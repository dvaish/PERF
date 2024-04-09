import pytest
from bitstring import Bits

from tidalsim.embeddings.markov_predictor import *

# Test consecutive cases here and in driver
class TestMarkovPredictor:
  def test_create_update_predict(self) -> None:
    predictor = markov_predictor(2)
    
    assert predictor.pattern_seen(Bits('0b00')) == False
    assert predictor.pattern_seen(Bits('0b01')) == False
    assert predictor.pattern_seen(Bits('0b10')) == False
    assert predictor.pattern_seen(Bits('0b11')) == False
    
    predictor.update(Bits('0b00'), Bits('0b0'))
    assert predictor.pattern_seen(Bits('0b00')) == True 
    # We haven't added 0b01 to the states table yet, as we don't have any outcomes for it
    assert predictor.pattern_seen(Bits('0b01')) == False
    
    predictor.update(Bits('0b01'), Bits('0b0'))
    assert predictor.pattern_seen(Bits('0b01')) == True 
    assert predictor.states[Bits('0b01')].freqs[Bits('0b0')] == 1
    assert predictor.states[Bits('0b01')].freqs[Bits('0b1')] == 0
    assert predictor.predict(Bits('0b01')) == Bits('0b0')
    
    predictor.update(Bits('0b11'), Bits('0b1'))

    predictor.update(Bits('0b01'), Bits('0b1'))
    predictor.update(Bits('0b01'), Bits('0b0'))
    assert predictor.predict(Bits('0b01')) == Bits('0b0')
    
    predictor.update(Bits('0b10'), Bits('0b1'))
    assert predictor.predict(Bits('0b10')) == Bits('0b1')
