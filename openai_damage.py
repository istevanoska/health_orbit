import base64
from openai import OpenAI

client = OpenAI()

# Многу едноставна шема: 0..3 (intact..collapsed)
JSON_SCHEMA = {
    "name": "damage_label",
    "schema": {
        "type": "object",
        "properties": {
            "damage_level": {"type": "integer", "minimum": 0, "maximum": 3},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "notes": {"type": "string"}
        },
        "required": ["damage_level", "confidence"],
        "additionalProperties": False
    }
}

def classify_damage_image_bytes(image_bytes: bytes) -> str:
    """
    Враќа JSON string по шема (damage_level/confidence/notes).
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    resp = client.responses.create(
        model="gpt-4o-mini",  # можеш и друг vision-capable модел
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": (
                    "You classify building damage from an image tile. "
                    "Return ONLY valid JSON following the given schema. "
                    "damage_level: 0=intact, 1=minor, 2=major, 3=collapsed."
                 )},
                {"type": "input_image", "image_url": data_url}
            ]
        }],
        text={"format": {"type": "json_schema", "json_schema": JSON_SCHEMA}}
    )

    return resp.output_text
