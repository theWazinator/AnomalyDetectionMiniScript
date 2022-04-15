"""
 Export clean/non-censored domain_ASes based on majority vote
 """
import sys
import json
import time
from itertools import islice
from multiprocessing import Manager, Process
import binascii
import os

cpus = 60  # change this param depend on how many CPUS you have/want to use


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)


def int2hex(final_len, _int):
    s = '%x' % _int
    _hex = binascii.unhexlify('0' + s if len(s) % 2 else s)
    if len(_hex) == final_len:
        return _hex
    else:
        return b'\x00' * (final_len - len(_hex)) + _hex


def worker(_file, start_line, end_line, DOMAIN_ASes, DOMAIN_IPs):
    local_DOMAIN_ASes = dict()
    local_DOMAIN_IPs = dict()
    with open(_file, 'r') as fo:
        for line in islice(fo, start_line, end_line, None):
            try:
                probe = json.loads(line.strip())
                try:
                    if probe['passed_liveness'] and \
                            probe['connect_error'] is False and \
                            probe['anomaly'] is False:
                        domain = probe['test_url']
                        for _dict in probe['response']:
                            if _dict['has_type_a'] is True:
                                responses = _dict['response']
                                for ip in responses:
                                    # AS
                                    asn = responses[ip]['asnum']
                                    if domain not in local_DOMAIN_ASes:
                                        local_DOMAIN_ASes[domain] = {asn: 1}
                                    else:
                                        if asn not in local_DOMAIN_ASes[domain]:
                                            local_DOMAIN_ASes[domain][asn] = 1
                                        else:
                                            local_DOMAIN_ASes[domain][asn] += 1
                                    # IP
                                    if domain not in local_DOMAIN_IPs:
                                        local_DOMAIN_IPs[domain] = {ip: 1}
                                    else:
                                        if ip not in local_DOMAIN_IPs[domain]:
                                            local_DOMAIN_IPs[domain][ip] = 1
                                        else:
                                            local_DOMAIN_IPs[domain][ip] += 1
                except KeyError:
                    continue
            except json.decoder.JSONDecodeError:
                continue
    for domain in local_DOMAIN_ASes:
        if domain not in DOMAIN_ASes:
            DOMAIN_ASes[domain] = local_DOMAIN_ASes[domain]
        else:
            for asn in local_DOMAIN_ASes[domain]:
                if asn not in DOMAIN_ASes[domain]:
                    DOMAIN_ASes[domain][asn] = local_DOMAIN_ASes[domain][asn]
                else:
                    DOMAIN_ASes[domain][asn] += local_DOMAIN_ASes[domain][asn]

    for domain in local_DOMAIN_IPs:
        if domain not in DOMAIN_IPs:
            DOMAIN_IPs[domain] = local_DOMAIN_IPs[domain]
        else:
            for ip in local_DOMAIN_IPs[domain]:
                if ip not in DOMAIN_IPs[domain]:
                    DOMAIN_IPs[domain][ip] = local_DOMAIN_IPs[domain][ip]
                else:
                    DOMAIN_IPs[domain][ip] += local_DOMAIN_IPs[domain][ip]
    return


def main():
    t0 = time.time()
    manager = Manager()
    DOMAIN_ASes = manager.dict()  # dict of domains and ASes they are mapped to based on Satellite
    DOMAIN_IPs = manager.dict()  # dict of domains and ASes they are mapped to based on Satellite

    _file = '/data/censorship/censored-planet/CP_Satellite-2022-02-16-12-00-01/results.json'
    lines = rawgencount(_file)
    print('Total {} lines'.format(lines))
    chunk_size = int(lines / cpus) + 1
    ps = list()

    for i in range(0, cpus):
        # print(i, tmp_sni_file, i*chunk_size, (i+1)*chunk_size,)
        p = Process(target=worker,
                    args=(_file,
                          i * chunk_size, (i + 1) * chunk_size,
                          DOMAIN_ASes, DOMAIN_IPs,))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    with open(_file.replace('/results.json', '/domain_ASes.csv'), 'w') as fo:
        for domain in DOMAIN_ASes:
            fo.write('{}|{}\n'.format(domain, DOMAIN_ASes[domain]))
    os.system('chown :cdac {}'.format(_file.replace('/results.json', '/domain_ASes.csv')))

    with open(_file.replace('/results.json', '/domain_IPs.csv'), 'w') as fo:
        for domain in DOMAIN_IPs:
            fo.write('{}|{}\n'.format(domain, DOMAIN_IPs[domain]))
    os.system('chown :cdac {}'.format(_file.replace('/results.json', '/domain_IPs.csv')))
    print('Done exporting {} domains in {}s'.format(len(DOMAIN_IPs), time.time() - t0))

main()