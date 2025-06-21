#!/usr/bin/env bash
accelerate launch --deepspeed ds_z3_A16.json \
       -m seal.loop task=arc
