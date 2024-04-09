import pytest
from bitstring import Bits

from tidalsim.embeddings.ppm_predictor import *

# Test consecutive cases here and in driver
class TestPPMPredictor:
  def test_create_update_predict(self) -> None:
    predictor = ppm_predictor(3)

    test_stream = Bits('0b011011101')

    i = 0

    assert len(list(predictor.predictors[0].states.keys())) == 1
    assert list(predictor.predictors[0].states.keys())[0] == Bits()
    assert predictor.predictors[0].pattern_seen(Bits()) == True 
    assert predictor.lowest_used_order == -1

    # 0-order predictor can predict either as correct in this case
    assert predictor.predict() == Bits('0b0') or predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 0
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[0:1]
    assert predictor.predictors[0].states[Bits()].freqs[Bits('0b0')] == 1
    assert predictor.predictors[0].states[Bits()].freqs[Bits('0b1')] == 0

    i = 1
    assert predictor.predict() == Bits('0b0')
    assert predictor.lowest_used_order == 0
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[0:2] 

    i = 2
    # 0-order predictor can predict either as correct in this case
    assert predictor.predict() == Bits('0b0') or predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 0
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]

    i = 3
    assert predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 1
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]
    
    i = 4
    assert predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 1
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]

    i = 5
    assert predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 2
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]

    i = 6
    assert predictor.predict() == Bits('0b0')
    assert predictor.lowest_used_order == 3
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]

    i = 7
    # Seen 11 -> 0 and 11 -> 1 in 2nd order predictor
    assert predictor.predict() == Bits('0b0') or predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 2
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]

    i = 8
    assert predictor.predict() == Bits('0b1')
    assert predictor.lowest_used_order == 3
    predictor.update(test_stream[i:i+1])
    assert predictor.curr_pattern == test_stream[i-2:i+1]