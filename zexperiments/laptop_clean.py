import os
import pandas as pd

data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "laptops.csv"), encoding="latin-1"
)
print(data.head())
print(data.columns)


def screenfilter(x):
    splits = x.strip().split()
    if len(splits) > 1:
        return " ".join(splits[:-1])
    else:
        return "Unknown"


def screenresx(x):
    splits = x.strip().split()
    return int(splits[-1].split("x")[0])


def screenresy(x):
    splits = x.strip().split()
    return int(splits[-1].split("x")[1])


data["ScreenType"] = data["ScreenResolution"].apply(screenfilter)
data["ScreenResolutionX"] = data["ScreenResolution"].apply(screenresx)
data["ScreenResolutionY"] = data["ScreenResolution"].apply(screenresy)
for i, j, k in zip(
    data["ScreenResolution"], data["ScreenResolutionX"], data["ScreenResolutionY"]
):
    # print(i, "@@@", j, "x", k)
    pass

data.rename(columns={"ScreenResolution": "OldFeature_ScreenResolution"}, inplace=True)


# print(data["ScreenType"].unique())
# print(data["ScreenResolutionX"].unique())
# print(data["ScreenResolutionY"].unique())
# print(data["Cpu"])


def cputype(x):
    splits = x.strip().split()
    if len(splits) > 1:
        return " ".join(splits[1:-1])
    else:
        raise ValueError("Unknown")


def cpubrand(x):
    splits = x.strip().split()
    if len(splits) > 1:
        return splits[0]
    else:
        raise ValueError("Unknown")


def cpupower(x):
    splits = x.strip().split()
    return float(splits[-1].replace("GHz", ""))


data["CpuBrand"] = data["Cpu"].apply(cpubrand)
data["CpuType"] = data["Cpu"].apply(cputype)
data["CpuPower"] = data["Cpu"].apply(cpupower)
data.rename(columns={"Cpu": "OldFeature_Cpu"}, inplace=True)

# print(data["CpuType"].unique())
# print(data["CpuBrand"].unique())
# print(data["CpuPower"].unique())


def ssd(x):
    splits = x.strip().split("+")
    for s in splits:
        if "SSD" in s:
            return float(s.split()[0].replace("GB", "").replace("TB", "000"))
    return 0


def hdd(x):
    splits = x.strip().split("+")
    for s in splits:
        if "HDD" in s:
            if "GB" in s:
                return float(s.split()[0].replace("GB", ""))
            elif "TB" in s:
                return float(s.split()[0].replace("TB", "")) * 1000
    return 0


def flash(x):
    splits = x.strip().split("+")
    for s in splits:
        if "Flash Storage" in s:
            if "GB" in s:
                return float(s.split()[0].replace("GB", ""))
            elif "TB" in s:
                return float(s.split()[0].replace("TB", "")) * 1000
    return 0


data["Memory_SSD (GB)"] = data["Memory"].apply(ssd)
data["Memory_HDD (GB)"] = data["Memory"].apply(hdd)
data["Memory_Flash (GB)"] = data["Memory"].apply(flash)
data.rename(columns={"Memory": "OldFeature_Memory"}, inplace=True)

# print(data["Memory_SSD (GB)"].unique())
# print(data["Memory_HDD (GB)"].unique())
# print(data["Memory_Flash (GB)"].unique())

data["Weight"] = data["Weight"].apply(lambda x: float(x.replace("kg", "")))
data["Price_euros"] = data["Price_euros"].apply(float)

print(data.columns)
useful_columns = [
    "Company",
    "TypeName",
    "Inches",
    "Ram",
    "OpSys",
    "Weight",
    "ScreenResolutionX",
    "ScreenResolutionY",
    "CpuBrand",
    "CpuPower",
    "Memory_SSD (GB)",
    "Memory_HDD (GB)",
    "Memory_Flash (GB)",
    "Price_euros",
]

data["Ram"] = data["Ram"].apply(lambda x: float(x.replace("GB", "")))
for col in useful_columns:
    print(col, data[col].unique() if len(data[col].unique()) < 20 else "")

data[useful_columns].to_csv(
    os.path.join(os.path.dirname(__file__), "laptops_clean.csv"), index=None
)
