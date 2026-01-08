#!/bin/bash
cd /home/ubuntu/gridworks-backoffice
source .venv/bin/activate
cd populate_database
python add_hourly_data.py >> /home/ubuntu/gridworks-backoffice/populate_database/cron.log 2>&1
python populate_database/add_house_params.py >> /home/ubuntu/gridworks-backoffice/populate_database/cron.log 2>&1

