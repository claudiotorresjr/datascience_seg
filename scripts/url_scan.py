import os

check_url_path = "/home/claudio/C3SL/projeto-corvus/corvus-url-reputation/check_url.sh"
check_url_conf = "/home/claudio/C3SL/projeto-corvus/corvus-url-reputation//data/templates/corvus.conf.template"

with open("./unknow_urls.txt", "r") as f:
    for line in f.readlines():
        real_url = os.popen(
            f"curl -s -o /dev/null --head -w '%{{url_effective}}\n' -L {line}").read()
        
        print(os.popen(f"{check_url_path} {check_url_conf} {real_url}").read())