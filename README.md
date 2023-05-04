## Background


* This repo provides deep-learning methods for myocardial T1 mapping.

### Usage


```
>>tigermyo -B c:\testB\ -BM c:\testB_MOCO\ -i 1 -l -o c:\result

We only provide specific formats(.dcm)
-B: Path to the input basal section of MOLLI RAW data
-M: Path to the input mid section of MOLLI RAW data
-A: Path to the input apical section of MOLLI RAW data
-BM: Path to the input basal section of MOLLI MOCO data
-MM: Path to the input mid section of MOLLI MOCO data
-AM: Path to the input apical section of MOLLI MOCO data
-only: Only use a certain method to calculate (1:input 2:LMT 3:VMTX)
-i: Iterative VMTX registration times
-l: Take the middle layer of LVM as output
-o: Path to output ALL results, if there is no value will only display the figure
```

### As a python package
```
pip install *
```
```
import tigermyo

config = {'input_B':"c:\test_B\",
          'input_B_MOCO':"c:\test_B_MOCO\",
          'input_M':"c:\test_M\",
          'input_M_MOCO':"c:\test_M_MOCO\",
          'input_A':"c:\test_A\",
          'input_A_MOCO':"c:\test_A_MOCO\",
          'only_use': 3,
          'iteration_VMTX': 4,
          'layering_mode': True,
          'output': "c:\result"}

tigermyo.run(config)
```

## Citation

* If you use this application, cite the following paper:
