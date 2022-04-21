methods = ['field_mean', 'carry_forward', 'carry_backward', 'carry_average', 'zero_fill', 'regression_sgd', 'regression_rand_forest', 'gan', 'knn', 'ms_gan']
sites = ['sttr001', 'sttr006', 'sttr008', 'sttr009', 'sttr010']
missing = ['10', '20', '30']
gaps = ['10', '600', '36000']

for m in methods:
    for s in sites:
        for p in missing:
            for g in gaps:
                for i in range(3):
                    msg = 'time python -u mink.py --method={} '.format(m)
                    msg += '--imputedata=data/{}.test.{}.{}.{} '.format(s, p, g, i)
                    msg += '--fulldata=data/{}.test '.format(s)
                    msg += '--ignorefile=data/{}.test.ignore '.format(s)
                    msg += '--spacing=0.1 '
                    msg += '--seqlen=64 '
                    msg += '--model-id={}.20.train '.format(s)
                    msg += '2>&1 | tee -a log/{}.{}.20.{}.{}.{}.eval'.format(m, s, p, g, i)
                    print(msg)
