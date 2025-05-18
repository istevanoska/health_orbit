from sentinelhub import SentinelHubRequest, SHConfig, DataCollection, MimeType, bbox_to_dimensions, BBox
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Configure Sentinel Hub using env variables
config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
config.instance_id = os.getenv("SH_INSTANCE_ID")  # Optional if not using custom instance

# ✅ Define bounding box and resolution
bbox = BBox(bbox=[21.40, 41.97, 21.45, 42.00], crs="EPSG:4326")
resolution = 10  # meters per pixel

# ✅ Create request
request = SentinelHubRequest(
    evalscript="""
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B03", "B02"],
        output: { bands: 3 }
      };
    }

    function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02];
    }
    """,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2024-05-01", "2024-05-17")
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=bbox,
    size=bbox_to_dimensions(bbox, resolution),
    config=config  # ✅ Pass the loaded config
)

# ✅ Get image
image = request.get_data()[0]  # NumPy array with shape (H, W, 3)

# Optional: Save or display image using matplotlib or OpenCV
