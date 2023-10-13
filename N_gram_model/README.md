# Complete the Sentence Using Markov Test Generation
### The mtg.py file is my NLP porject python file and test_mtg.py is my testing python file

**Description:**

Write a bare-bones Markov text generator. Implement a function of the form `''
finish_sentence(sentence, n, corpus, randomize=False)
`` ` 
that takes four arguments:

- a sentence [list of tokens] that we're trying to build on,
- n [int], the length of n-grams to use for prediction, and
- a source corpus [list of tokens]
- a flag indicating whether the process should be randomized [bool]

The function returns an extended sentence until the first ., ?, or ! is found OR until it has 10 total tokens.

If the input flag `randomize` is false, choose at each step the single most probable next token. When two tokens are equally probable, choose the one that occurs first in the corpus. This is called a deterministic process.

If `randomize` is true, draw the next word randomly from the appropriate distribution.

Use stupid backoff ( ) and no smoothing.

Provide some example applications of your function in both deterministic and stochastic modes, for a few sets of seed words and a few different n.

As one (simple) test case, use the following inputs.

#### The following contains potential test case sentences and their respective finished sentence output

## Deterministic Method:
1. Given sentence = ['robot'] n=3 randomize = False
  Output: ['robot', ',', 'and', 'the', 'two', 'miss', 'steeles', ',', 'as', 'she']

2. Given sentence = ['she', 'was', 'not'] n=1 randomize = False
   Output: ['she', 'was', 'not', ',', ',', ',', ',', ',', ',', ',']
   
3. Given sentence = ['robot'] n=2 randomize = False
   Output: ['robot', ',', 'and', 'the', 'same', 'time', ',', 'and', 'the', 'same']
  
4. Given sentence = ['she', 'was', 'not'] n=3 randomize = False
  Output: ['she', 'was', 'not', 'in', 'the', 'world', '.']

## Stochastic Method
1. Given sentence = ['robot'] n=3 randomize = True
   ['robot', 'own', 'horses', ',', 'and', 'should', 'it', 'ever', 'be', 'tolerably']
   
2. Given sentence = ['she', 'was', 'not'] n=1 randomize = True
   ['she', 'was', 'not', 'minutes', 'to', 'their', 'which', 'much', ',', 'had']
   
3. Given sentence = ['robot'] n=2 randomize = True
   ['robot', 'talk', 'of', 'his', 'death', ';', 'for', 'after', 'staying', 'three']
   
4. Given sentence = Given sentence = ['she', 'was', 'not'] n=3 randomize = True
   Output: ['she', 'was', 'not', 'fortunate', 'enough', 'to', 'find', 'out', 'his', 'hand']








