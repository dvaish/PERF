from bitstring import BitArray, Bits, BitStream
from typing import List, Union
from dataclasses import dataclass, field
from tidalsim.embeddings.markov_predictor import markov_predictor, state

# Implement update exclusion => update used predictor and higher predictors

@dataclass
class ppm_predictor:
  order: int
  curr_pattern: BitArray = field(default_factory=BitArray)
  # last_seen: BitArray = field(default=BitArray)
  lowest_used_order: int = field(default=-1)
  predictors: List[markov_predictor] = field(default_factory=list)

  def __post_init__(self):
    for i in range(self.order + 1):
      self.predictors.append(markov_predictor(i))
      
    # Initialize 0-order predictor
    self.predictors[0].states[Bits()] = state()

  def predict(self):

    self.lowest_used_order = -1
    for i in range(self.curr_pattern.len, 0 - 1, -1):
      if self.predictors[i].pattern_seen(self.curr_pattern[self.curr_pattern.len - i:]):
        self.lowest_used_order = i
        return self.predictors[i].predict(self.curr_pattern[self.curr_pattern.len - i:])


  def update(self, outcome: Union[Bits, BitArray]) -> None:
    assert outcome.len == 1

    for i in range(self.curr_pattern.len, self.lowest_used_order - 1, -1):
      self.predictors[i].update(self.curr_pattern[self.curr_pattern.len - i:], outcome)

    self.curr_pattern.append(outcome)
    if self.curr_pattern.len > self.order:
      self.curr_pattern = self.curr_pattern[self.curr_pattern.len - self.order:] 
