import os
import torch


class FileBasedLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    A PyTorch learning rate scheduler that reads the learning rate from a file at each step.
    The file should contain a single float value representing the learning rate.
    """

    def __init__(self, optimizer, file_path, default_lr=1e-9, last_epoch=-1):
        self.file_path = file_path
        self.last_modified_time = None
        self.cached_lr = None
        self.default_lr = default_lr
        super().__init__(optimizer, last_epoch)
        self.read_lr_from_file()  # Initial read to set up the scheduler

    def read_lr_from_file(self):
        """
        Read the learning rate from the file and update the last modified time.
        """
        with open(self.file_path, "r") as file:
            lr = float(file.read().strip())

        # Check if the read learning rate is valid
        if lr <= 0:
            raise ValueError(
                f"The learning rate read from the file is not positive: {lr}"
            )

        # Update the cached learning rate and the last modified time
        self.cached_lr = lr
        self.last_modified_time = os.path.getmtime(self.file_path)

    def get_lr(self):
        """
        Efficiently get the learning rate by checking the file modification time.
        """
        if not os.path.isfile(self.file_path):
            # Write the learning rate to the specified file
            with open(file_path, "w") as file:
                file.write(self.default_lr)
            print(
                f"The learning rate {self.default_lr} has been written to {file_path}."
            )

        # Check if the file was modified since the last read
        current_modified_time = os.path.getmtime(self.file_path)
        if (
            self.last_modified_time is None
            or current_modified_time > self.last_modified_time
        ):
            self.read_lr_from_file()

        # Return the cached learning rate for all parameter groups
        return [self.cached_lr for _ in self.optimizer.param_groups]


def continuously_write_learning_rate_to_file(file_path):
    while True:
        try:
            # Prompt the user for a learning rate
            lr = input("Please enter the new learning rate (or 'exit' to quit): ")

            # Check if the user wants to exit the loop
            if lr.lower() == "exit":
                print("Exiting the learning rate update loop.")
                break

            # Convert the input to a float and validate
            float_lr = float(lr)
            if float_lr <= 0:
                print("The learning rate must be a positive number.")
            else:
                # Write the learning rate to the specified file
                with open(file_path, "w") as file:
                    file.write(lr)
                print(f"The learning rate {lr} has been written to {file_path}.")
        except ValueError:
            print("Invalid input. Please enter a valid number for the learning rate.")


# The file path where the learning rate is stored
file_path = "lr_value.txt"  # Replace with the path to your learning rate file

if __name__ == "__main__":
    # Call the function to enter the loop
    continuously_write_learning_rate_to_file(file_path)
