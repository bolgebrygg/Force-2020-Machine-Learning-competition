import unittest
from pathlib import Path

import pandas as pd

from main import main


class MainTestCase(unittest.TestCase):
    _test_pred_path = Path(__file__).parent.joinpath('test_predictions.csv')

    def setUp(self) -> None:
        if self._test_pred_path.exists():
            self._test_pred_path.unlink()

    def test_results(self):
        main(
            input_csv=Path(__file__).parent.joinpath('test_data.csv'),
            output_csv=self._test_pred_path
        )

        reference_pred = pd.read_csv(Path(__file__).parent.joinpath('SubmissionEnsamble_v1.csv'))
        test_pred = pd.read_csv(self._test_pred_path)

        num_different_predictions = sum((reference_pred['lithology'] != test_pred['lithology']))
        print(
            f'Found {num_different_predictions} mismatch in predictions ({num_different_predictions / len(reference_pred) * 100:.2f}%)')

        self.assertLess(num_different_predictions, 100)


if __name__ == '__main__':
    unittest.main()
