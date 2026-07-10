# Proxy CPT ranking (auto-generated)

Source: `data/proxy_results.csv` (26 weight combinations). Regenerate with `python3 analysis/proxy_cpt_ranking.py`.

**Selected** (`data/best_weight.json`): `W11` (Cultural-heavy) — macro_loss=2.1741, worst_language_loss=3.3758, breaking a 16-way tie (all within 1% of the lowest macro_loss) by worst_language_loss.

| rank | weight_id | meaning | macro_loss | worst_language_loss | tie group? | selected |
|---:|---|---|---:|---:|:---:|:---:|
| 1 | W04 | Cleanliness only | 2.1644 | 3.3816 | Y |  |
| 2 | W05 | Cultural only | 2.1687 | 3.3815 | Y |  |
| 3 | W21 | Cleanliness + cultural | 2.1725 | 3.3833 | Y |  |
| 4 | W11 | Cultural-heavy | 2.1741 | 3.3758 | Y | **Y** |
| 5 | W10 | Cleanliness-heavy | 2.1760 | 3.4038 | Y |  |
| 6 | W24 | Reasoning + cleanliness + cultural | 2.1778 | 3.3965 | Y |  |
| 7 | W23 | Edu + cleanliness + cultural | 2.1785 | 3.3850 | Y |  |
| 8 | W15 | Edu + cultural | 2.1792 | 3.3919 | Y |  |
| 9 | W17 | Reasoning + cleanliness | 2.1816 | 3.3998 | Y |  |
| 10 | W18 | Reasoning + cultural | 2.1818 | 3.4044 | Y |  |
| 11 | W02 | Reasoning only | 2.1820 | 3.4169 | Y |  |
| 12 | W26 | Professionalism + cleanliness + cultural | 2.1822 | 3.4027 | Y |  |
| 13 | W06 | Equal average | 2.1832 | 3.4039 | Y |  |
| 14 | W25 | Edu + reasoning + cleanliness | 2.1836 | 3.4067 | Y |  |
| 15 | W19 | Professionalism + cleanliness | 2.1837 | 3.4163 | Y |  |
| 16 | W14 | Edu + cleanliness | 2.1855 | 3.4156 | Y |  |
| 17 | W07 | Edu-heavy | 2.1863 | 3.4123 |  |  |
| 18 | W16 | Reasoning + professionalism | 2.1863 | 3.4166 |  |  |
| 19 | W20 | Professionalism + cultural | 2.1868 | 3.4072 |  |  |
| 20 | W12 | Edu + reasoning | 2.1875 | 3.4206 |  |  |
| 21 | W13 | Edu + professionalism | 2.1879 | 3.4155 |  |  |
| 22 | W08 | Reasoning-heavy | 2.1881 | 3.4190 |  |  |
| 23 | W22 | Edu + reasoning + professionalism | 2.1885 | 3.4188 |  |  |
| 24 | W09 | Professionalism-heavy | 2.1896 | 3.4309 |  |  |
| 25 | W01 | Edu only | 2.1903 | 3.4137 |  |  |
| 26 | W03 | Professionalism only | 2.2211 | 3.4699 |  |  |

