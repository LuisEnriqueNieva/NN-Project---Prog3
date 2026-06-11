#!/usr/bin/env bash

project_name='prog3_pf_epic1_feature4_v2026_1'
source_code='
  include/utec
  src/feature4_solution.cpp
  '
cd .. || exit 1
rm -f ${project_name}.zip
zip -r -S ${project_name} ${source_code}
