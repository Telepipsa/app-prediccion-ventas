from holidays import Spain

if __name__ == '__main__':
    h = Spain(years=[2025])
    for d, name in sorted(h.items()):
        print(d, name)
