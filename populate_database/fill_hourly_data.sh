#!/bin/bash
cd /home/ubuntu/gridworks-backoffice/populate_database
source .venv/bin/activate
python add_hourly_data.py >> /home/ubuntu/gridworks-backoffice/populate_database/cron.log 2>&1

