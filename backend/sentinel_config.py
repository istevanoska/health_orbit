# import os
# from sentinelhub import SHConfig
# from dotenv import load_dotenv
#
# load_dotenv()  # Load env variables from .env
#
# def get_sentinel_config():
#     config = SHConfig()
#     config.instance_id = os.getenv('SH_INSTANCE_ID')
#     config.sh_client_id = os.getenv('SH_CLIENT_ID')
#     config.sh_client_secret = os.getenv('SH_CLIENT_SECRET')
#
#     if not config.sh_client_id or not config.sh_client_secret:
#         raise ValueError("Missing SH_CLIENT_ID or SH_CLIENT_SECRET in .env")
#
#     return config
