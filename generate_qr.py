"""
generate_qr.py  —  Run this after deploying to Streamlit Cloud.

Usage:
    python generate_qr.py https://your-app-name.streamlit.app

Output:
    qrcode.png  (scan with phone to open the deployed app)
"""
import sys
try:
    import qrcode
except ImportError:
    print("Install qrcode first:  pip install qrcode[pil]")
    sys.exit(1)

url = sys.argv[1] if len(sys.argv) > 1 else "https://your-app-name.streamlit.app"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=12,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="#1a1a2e", back_color="white")
out = "qrcode.png"
img.save(out)
print(f"Saved: {out}")
print(f"URL:   {url}")
