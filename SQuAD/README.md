# InfoBERT on SQuAD

## Train

Running standard SQuAD training:
```bash
#                   [runstandardsquad]  [custom]   [gpu]   [bszie]   [beta]   [version]  [mname]
source setup.sh && runstandardsquad  squad-s       4         8       0         3    roberta-large 
```

Running InfoBERT for SQuAD:

```bash
##                  runsquad        [custom]    [gpu] [bszie]  [beta]   [version]  [hdp]  [adp]  [alr]  [amag]  [anorm]  [asteps]          [mname]  [alpha] [cl] [ch]
source setup.sh && runsquad         roberta      4      8      5e-5          6     0.1      0.1    2e-2  2e-2   4e-2         2         roberta-large  5e-3  0.75  0.95 
```

## Evaluate

Run standard evaluation for SQuAD models

```bash
##                  [evalsquad]  [custom]   [gpu]   [bszie]   [beta]   [version]  [hdp]  [adp]  [alr]  [amag]  [anorm]  [asteps]  [mname]
#source setup.sh &&  evalsquad     squad     4         10      0            3       0.1   0       4e-2  8e-2    0          3    squad-bert-large-uncased-whole-word-masking-sl384-lr3e-5-bs6-beta0-alr4e-2-amag8e-2-anm0-as3-hdp0.1-adp0-version3
```