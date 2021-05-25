from os import makedirs
import requests


def confirm(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def download(id, path):
    base_url = 'https://drive.google.com/uc?export=download'

    session = requests.Session()
    res = session.get(base_url, params={'id': id}, stream=True)

    token = confirm(res)
    if token:
        res = session.get(base_url, params={'id': id, 'confirm': token}, stream=True)

    with open(path, 'wb') as f:
        MAX_ITER = 10000
        for content in res.iter_content(MAX_ITER):
            if content:
                f.write(content)


if __name__ == '__main__':
    dis_id = '1RKNIU6VWsDRcb0AiCWjEMrtTlhJCx_Hs'
    gen_id = '1k9WSqT8H2t5WSO_dfaERrA90N4R_nehO'

    dest_dir = 'pretrained'
    makedirs(dest_dir, exist_ok=True)

    dis_path = dest_dir + '/dis.npz'
    gen_path = dest_dir + '/gen.npz'


    print('[Message] Downloading pre trained weight (usually it may take few minutes)')

    # Discriminator
    print('[Message] Now downloading Discriminator\'s weight... (about 5MB)')
    download(dis_id, dis_path)

    # Generator
    print('[Message] Now downloading Generator\'s weight... (about 200MB)')
    download(gen_id, gen_path)

    print('[Message] Successfully')
