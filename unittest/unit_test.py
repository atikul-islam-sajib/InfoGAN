import unittest
import sys
import os
import joblib as pkl

sys.path.append("src/")

from config import PROCESSED_PATH
from dataloader import Loader


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.dataloader = pkl.load(os.path.join(PROCESSED_PATH, "dataloader.pkl"))

    def test_quantity_data(self):
        self.assertEqual(Loader(batch_size=128).quantity_data(), 60000)


if __name__ == "__main__":
    unittest.main()
