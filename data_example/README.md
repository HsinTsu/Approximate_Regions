## Data

Large event logs and evaluation datasets are not included in this repository.

Small toy example logs are provided in:

```text
data_example/
```

These logs are used for testing and illustrating the behavior of the discovery pipeline under simple control-flow patterns and controlled noise settings.


### Choice_Merge_case: Trace variants: {"ab", "ac", "db"}
```text
Choice_Merge_case1.xes    # No noise
Choice_Merge_case2.xes    # Variant "ac" is treated as noise
Choice_Merge_case4.xes    # Variant "ab" is treated as noise
Choice_Merge_case5.xes    # Variant "db" is treated as noise
```


### Loop_case: Trace variants: {"abc", "abbc"}
```text
Loop_case1.xes            # No noise
Loop_case2.xes            # Variant "abc" is treated as noise
Loop_case3.xes            # Variant "abbc" is treated as noise
```


### Skip_case: Trace variants: {"abc", "ac"}
```text
Skip_case1.xes            # No noise
Skip_case2.xes            # Variant "ac" is treated as noise
Skip_case3.xes            # Variant "abc" is treated as noise
```


### Sepsis Cases
```text
Sepsis Cases - ER only.xes                              # ER activities filtered from Sepsis Cases
Sepsis Cases - tests only.xes                           # test activities filtered from Sepsis Cases
```


### Road_Traffic_Fine_Management_Process
```text
Road_Traffic_Fine_Management_Process - sampled.xes      # 2000 sampled traces from Road_Traffic_Fine_Management_Process
```
