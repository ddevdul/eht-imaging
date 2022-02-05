import sys
sys.path.append("../")
import array
from unittest import TestCase, main

class ArrayTestClass(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass
    
    def test_basic(self):
        self.assertTrue(True)
    
    @classmethod
    def tearDownClass(cls) -> None:
        pass

if __name__ == "__main__":
    main()