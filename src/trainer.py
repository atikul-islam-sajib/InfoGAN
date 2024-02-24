import sys
import os
import logging
import argparse
import torch
import joblib as pkl
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
    filename="./logs/trainer.log",
)

from config import PROCESSED_PATH, MODELS_CHECKPOINTS, BEST_MODEL_PATH, SAVE_IMAGE_PATH
from utils import device_init, weight_init
from discriminator import Discriminator
from generator import Generator
from QNet import QNet


class Trainer:
    def __init__(
        self,
        epochs=100,
        in_channels=1,
        lr=0.0002,
        latent_space=100,
        batch_size=128,
        beat1=0.5,
        device="mps",
        display=False,
    ):
        self.epochs = epochs
        self.in_channels = in_channels
        self.latent_space = latent_space
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beat1
        self.device = device_init(device=device)
        self.dataloader = self.load_data()
        self.display = display

        try:
            self.net_D, self.net_G, self.net_Q = self.define_models()
        except Exception as e:
            logging.info("Initialization cannot be possible".capitalize())
        else:
            self.net_D.apply(weight_init)
            self.net_G.apply(weight_init)
            self.net_Q.apply(weight_init)

            self.optimizer_D, self.optimizer_G, self.optimizer_Q = (
                self.define_optimizer(
                    net_D=self.net_D, net_G=self.net_G, net_Q=self.net_Q
                )
            )
            self.criterion_D = nn.BCELoss()
            self.criterion_Q = nn.CrossEntropyLoss()

    def define_models(self):
        discriminator = Discriminator(in_channels=self.in_channels).to(self.device)
        generator = Generator(latent_space=self.latent_space).to(self.device)
        qnet = QNet().to(self.device)

        return discriminator, generator, qnet

    def define_optimizer(self, **params):
        optimizer_D = optim.Adam(
            params["net_D"].parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        optimizer_G = optim.Adam(
            params["net_G"].parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        optimizer_Q = optim.Adam(
            params["net_Q"].parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        return optimizer_D, optimizer_G, optimizer_Q

    def load_data(self):
        if os.path.exists(os.path.join(PROCESSED_PATH, "dataloader.pkl")):
            dataloader = pkl.load(
                filename=os.path.join(PROCESSED_PATH, "dataloader.pkl")
            )
            return dataloader
        else:
            raise Exception("DataLoader cannot be loaded".capitalize())

    def train_discriminator(self, **params):
        self.optimizer_D.zero_grad()

        real_loss = self.criterion_D(
            self.net_D(params["real_image"]), params["real_labels"]
        )

        real_loss.backward(retain_graph=True)

        fake_loss = self.criterion_D(
            self.net_D(params["fake_image"]), params["fake_labels"]
        )

        fake_loss.backward(retain_graph=True)

        self.optimizer_D.step()

        return real_loss.item() + fake_loss.item()

    def train_generator(self, **params):
        self.optimizer_G.zero_grad()
        self.optimizer_Q.zero_grad()

        generated_samples = self.net_D(params["noise_samples"])
        generated_loss = self.criterion_D(generated_samples, params["real_labels"])

        labels = torch.randint(0, 10, (params["batch_size"],), dtype=torch.long).to(
            self.device
        )
        QNet_predict = self.net_Q(params["noise_samples"])
        QNet_loss = self.criterion_Q(QNet_predict, labels)

        total_G_loss = generated_loss + QNet_loss
        total_G_loss.backward(retain_graph=True)

        self.optimizer_G.step()
        self.optimizer_Q.step()

        return total_G_loss.item(), QNet_loss.item()

    def save_checkpoints(self, **kwargs):
        if kwargs["epoch"] != self.epochs:
            if os.path.exists(MODELS_CHECKPOINTS):
                torch.save(
                    self.net_G.state_dict(),
                    os.path.join(
                        MODELS_CHECKPOINTS, "G_{}.pth".format(kwargs["epoch"])
                    ),
                )
            else:
                raise Exception("Checkpoints directory not found".capitalize())
        else:
            if os.path.exists(BEST_MODEL_PATH):
                torch.save(
                    self.net_G.state_dict(), os.path.join(BEST_MODEL_PATH, "best_G.pth")
                )

    def train(self):

        for epoch in range(self.epochs):
            D_loss = list()
            G_loss = list()
            Q_loss = list()
            for index, (real_image, _) in enumerate(self.dataloader):
                real_image = real_image.to(self.device)
                batch_size = real_image.size(0)

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                noise_samples = torch.randn(batch_size, self.latent_space, 1, 1).to(
                    self.device
                )
                fake_images = self.net_G(noise_samples)

                d_loss = self.train_discriminator(
                    real_image=real_image,
                    fake_image=fake_images,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                )

                g_loss, q_loss = self.train_generator(
                    noise_samples=fake_images,
                    real_labels=real_labels,
                    batch_size=batch_size,
                )

                D_loss.append(d_loss)
                G_loss.append(g_loss)
                Q_loss.append(q_loss)

            try:
                self.save_checkpoints(epoch=epoch + 1)
            except Exception as e:
                print("Saving checkpoints is not possible".capitalize())
            else:
                if index % 5:
                    image = self.net_G(
                        torch.randn(20, self.latent_space, 1, 1).to(self.device)
                    )
                    save_image(
                        image,
                        os.path.join(SAVE_IMAGE_PATH, "image_{}.png".format(index)),
                        nrow=5,
                        normalize=True,
                    )

            try:
                if self.display:
                    print(
                        "Epochs - {}/{} [=======] g_loss: {} - d_loss: {} - q_loss: {}".format(
                            epoch + 1,
                            self.epochs,
                            np.mean(g_loss),
                            np.mean(d_loss),
                            np.mean(q_loss),
                        )
                    )
                else:
                    logging.info(
                        "Epochs - {}/{} [=======] g_loss: {} - d_loss: {} - q_loss: {}".format(
                            epoch + 1,
                            self.num_epochs,
                            np.mean(g_loss),
                            np.mean(d_loss),
                            np.mean(q_loss),
                        )
                    )
            except Exception as e:
                logging.info("Displaying is not possible".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer".title())
    parser.add_argument(
        "--epochs", type=int, default=10, help="Define the epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Define the learning rate".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Define the batch size".capitalize()
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space".capitalize(),
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=True,
        help="Define if you want to display or not".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Define the number of channels".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        if (
            args.device
            and args.epochs
            and args.lr
            and args.batch_size
            and args.latent_space
            and args.display
            and args.in_channels
        ):
            logging.info("Training the model".capitalize())

            trainer = Trainer(
                epochs=args.epochs,
                in_channels=args.in_channels,
                lr=args.lr,
                latent_space=args.latent_space,
                batch_size=args.batch_size,
                display=args.display,
                device=args.device,
            )
            trainer.train()

            logging.info("Training finished".capitalize())
        else:
            logging.exception("All arguments should be defined".capitalize())

    else:
        logging.exception("Training should be defined".capitalize())
