def data_exploration():
    import os

    decades = ['1800s', '1820s', '1840s', '1860s', '1880s',
               '1900s', '1920s', '1940s', '1960s']

    for decade in decades:
        path = f'data/raw/lit_{decade}.txt'
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            words = sum(len(l.split()) for l in lines)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"{decade}: {words:,} words | {size_mb:.1f} MB")
        else:
            print(f"{decade}: FILE NOT FOUND — re-run download script")