#!/bin/bash
# A script for running the docker container and experiements

mkdir -p $(pwd)/mlruns

mkdir -p $(pwd)/artifacts

docker run -v $(pwd)/mlruns:/Domain-Guided-Monitoring/mlruns \
-v $(pwd)/artifacts:/Domain-Guided-Monitoring/artifacts  \
--rm  \
--name domainml \
-td \
domainml

docker exec -d domainml sh -c "seq 50 | xargs -n 1 -I -- make -C /Domain-Guided-Monitoring/. run_huawei_causal size=-1 type=heuristic"

docker exec -d domainml sh -c "seq 50 | xargs -n 1 -I -- make -C /Domain-Guided-Monitoring/. run_huawei_causal size=-1 type=score"

docker exec -d domainml sh -c "seq 50 | xargs -n 1 -I -- make -C /Domain-Guided-Monitoring/. run_huawei_causal size=-1 type=constraint"

docker exec -d domainml sh -c "seq 50 | xargs -n 1 -I -- make -C /Domain-Guided-Monitoring/. run_huawei_simple size=-1"