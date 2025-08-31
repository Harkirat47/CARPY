import os
import requests

reference_urls = {
    0: [
        "https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg",
        "https://images.pexels.com/photos/733872/pexels-photo-733872.jpeg",
        "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg",
        "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg",
        "https://images.pexels.com/photos/928243/pexels-photo-928243.jpeg",
        "https://images.pexels.com/photos/91227/pexels-photo-91227.jpeg",
        "https://images.pexels.com/photos/1222271/pexels-photo-1222271.jpeg",
        "https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg",
        "https://images.pexels.com/photos/2380794/pexels-photo-2380794.jpeg",
        "https://images.pexels.com/photos/1102341/pexels-photo-1102341.jpeg"
    ],
    2: [
        "https://images.pexels.com/photos/358070/pexels-photo-358070.jpeg",
        "https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg",
        "https://images.pexels.com/photos/799443/pexels-photo-799443.jpeg",
        "https://images.pexels.com/photos/3860095/pexels-photo-3860095.jpeg",
        "https://images.pexels.com/photos/305070/pexels-photo-305070.jpeg",
        "https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg",
        "https://images.pexels.com/photos/1231643/pexels-photo-1231643.jpeg",
        "https://images.pexels.com/photos/210018/pexels-photo-210018.jpeg",
        "https://images.pexels.com/photos/2280948/pexels-photo-2280948.jpeg",
        "https://images.pexels.com/photos/1007415/pexels-photo-1007415.jpeg"
    ]
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

base_folder = "calibrator_refs"
os.makedirs(base_folder, exist_ok=True)

for cls, urls in reference_urls.items():
    folder = os.path.join(base_folder, str(cls))
    os.makedirs(folder, exist_ok=True)
    for idx, url in enumerate(urls):
        filename = os.path.join(folder, f"ref_{idx}.jpg")
        if not os.path.exists(filename):
            print(f"Downloading class {cls} image {idx}...")
            try:
                r = requests.get(url, headers=headers)
                with open(filename, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed to download {url} — {e}")

print(" Finished downloading reference images.")
