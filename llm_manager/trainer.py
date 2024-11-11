from typing import Optional, Union, List

import torch
from torch.nn import Parameter
import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from transformers.utils import is_datasets_available


class CustomTrainer(Trainer):
    """
    This custom trainer is needed, so we can have different data collators while training and evaluating.
    For that we adjust the get_eval_dataloader method.
    """

    def __init__(
        self,
        *args,
        eval_data_collator=None,
        gnn_parameters: Optional[List[Parameter]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.test_data_collator = eval_data_collator
        self.gnn_parameters = gnn_parameters

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.val_dataset[eval_dataset]` as the evaluation dataset.
                If a `Dataset`, will override `self.val_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`],
                columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])
        eval_dataset = (
            self.eval_dataset[eval_dataset]  # type: ignore
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )  # type: ignore
        data_collator = self.test_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )  # type: ignore
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator,  # type: ignore
                description="evaluation",
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):  # type: ignore
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)  # type: ignore
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)  # type: ignore
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}
        return self.accelerator.prepare(eval_dataloader)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        We adjust the create optimizer method, so we can add the
        """
        super().create_optimizer_and_scheduler(num_training_steps)

        # add gnn parameters if available
        if self.optimizer and self.gnn_parameters:
            # Add parameters of the other model to the optimizer
            self.optimizer.add_param_group({"params": self.gnn_parameters})  # type: ignore TODO: Fix typing issues
