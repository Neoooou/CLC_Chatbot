import unittest
from models import InformationRetrieval

class TestInformationRetrieval(unittest.TestCase):

    def test_init(self):
        ir = InformationRetrieval()

    def test_retrieve_name_correct(self):
        ir = InformationRetrieval()
        name = ir.retrieve_name('name is John')
        self.assertEqual(name, 'John')

    def test_retrieve_name_incorrect(self):
        ir = InformationRetrieval()
        name = ir.retrieve_name('this sentence has no name')
        self.assertEqual(name, None)
    
    def test_retrieve_place_correct(self):
        ir = InformationRetrieval()
        place = ir.retrieve_place('at Tesco')
        self.assertEqual(place, 'Tesco')

    def test_retrieve_place_incorrect(self):
        ir = InformationRetrieval()
        place = ir.retrieve_place('no place')
        self.assertEqual(place, None)

    def test_retrieve_time_correct(self):
        ir = InformationRetrieval()
        time = ir.retrieve_time('sometime at noon')
        self.assertEqual(time, 'noon')

    def test_retrieve_time_incorrect(self):
        ir = InformationRetrieval()
        time = ir.retrieve_time('no time')
        self.assertEqual(time, None)

    def test_retrieve_email_correct(self):
        ir = InformationRetrieval()
        email = ir.retrieve_email('this is some email email@email.com')
        self.assertEqual(email, 'email@email.com')

    def test_retrieve_email_incorrect(self):
        ir = InformationRetrieval()
        email = ir.retrieve_email('there was no email ever')
        self.assertEqual(email, None)

    def test_retrieve_phone_correct(self):
        ir = InformationRetrieval()
        phone = ir.retrieve_phone('my phone number is 07958143123')
        self.assertEqual(phone, '07958143123')

    def test_retrieve_phone_incorrect(self):
        ir = InformationRetrieval()
        phone = ir.retrieve_phone('there is no phone number')
        self.assertEqual(phone, None)

if __name__ == '__main__':
    unittest.main()