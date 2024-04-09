from bitstring import BitArray, Bits
from typing import Dict, Union 
from dataclasses import dataclass, field

class MarkovPredictorException(Exception):
  pass

@dataclass
class state:
  freqs: Dict[Bits, int] = field(default_factory=lambda: {Bits('0b0'): 0, Bits('0b1'): 0}) 

  def predict(self) -> Bits:
    # Can't predict on state with 0 frequencies.
    # assert max(self.freqs.values()) != 0
    return max(self.freqs, key=self.freqs.get)
  
  def update(self, next_bit: Bits) -> None:
    assert next_bit.len == 1
    self.freqs[next_bit] += 1

@dataclass
class markov_predictor:
  # order of predicter
  order: int
  # which entry do we update with the next outcome?
  curr_pattern: BitArray = field(default_factory=BitArray)
  # table of all patterns/outcomes seen so far
  states: Dict[Bits, state] = field(default_factory=dict)

  def pattern_seen(self, pattern: Union[Bits, BitArray]) -> bool:
    return Bits(pattern) in self.states

  def predict(self, pattern: Union[Bits, BitArray]) -> Bits:
    # Can't predict on unseen pattern
    assert self.pattern_seen(pattern)
    return self.states[Bits(pattern)].predict()
  
  def update(self, pattern: Union[Bits, BitArray], outcome: Union[Bits, BitArray]) -> None:
    assert outcome.len == 1
    assert pattern.len == self.order
    
    # if self.curr_pattern.len < self.len:
    #   # haven't seen enough to predict
    #   self.curr_pattern.append(outcome)
    #   return
    
    if not self.pattern_seen(pattern):
      self.states[Bits(pattern)] = state()
    
    self.states[Bits(pattern)].update(next_bit=Bits(outcome))
    
    # self.curr_pattern <<= 1 
    # self.curr_pattern[-1] = outcome