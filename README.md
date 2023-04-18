## Background


* This repo provides deep-learning methods for myocardial T1 mapping.

### Usage


```
>>tigermyo --input_B c:\testB\ --input_B_MOCO c:\testB_MOCO\ --layering_mode --output c:\result.png

We only provide specific formats(.dcm)
--input_B: Path to the input basal section of MOLLI RAW data
--input_M: Path to the input mid section of MOLLI RAW data
--input_A: Path to the input apical section of MOLLI RAW data
--input_B_MOCO: Path to the input basal section of MOLLI MOCO data
--input_M_MOCO: Path to the input mid section of MOLLI MOCO data
--input_A_MOCO: Path to the input apical section of MOLLI MOCO data
--only_use: Only use a certain method to calculate (1:input 2:LMT 3:VMTX)
--iteration_VMTX: Iterative VMTX registration times
--layering_mode: Take the middle layer of LVM as output
--output: Path to output AHA17 result,if there is no value will only display the figure
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
          'output': "c:\result.png"}

tigermyo.run(config)
```

## Citation

* If you use this application, cite the following paper:
