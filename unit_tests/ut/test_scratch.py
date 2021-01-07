import pytest
from mymul import only_odd_mul 
       
def test_odd_numbers():
   assert only_odd_mul(3, 5) == 15
   
def test_even_numbers():
    with pytest.raises(ValueError):
        only_odd_mul(2,4)

#pytest --cov=test_scratch