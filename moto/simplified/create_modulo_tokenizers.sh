#!/bin/bash
python -m moto.simplified.create_numeric_synthetic_data \
      hf.hf_repo_id=flexitok/mod-arithmetic  hf.publish_to_hf=true \
        a_max=99999 seed=42 \
        moduli=[2,3,4,5,7,8,10,16,25,32,100,125,1000] \
        num_shards=1 \
        split_by_problem_type=true
