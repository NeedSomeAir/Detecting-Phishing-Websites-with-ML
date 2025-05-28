# 📊 Dataset Files

Dataset files for phishing website detection project containing raw URLs and extracted features.

## 📁 File Descriptions

| File                            | Description                               | Count    | Source                                                       |
| ------------------------------- | ----------------------------------------- | -------- | ------------------------------------------------------------ |
| **1.Benign_list_big_final.csv** | Raw legitimate URLs                       | 35,300   | [UNB Dataset](https://www.unb.ca/cic/datasets/url-2016.html) |
| **2.online-valid.csv**          | Raw phishing URLs                         | Variable | [PhishTank](https://www.phishtank.com/developer_info.php)    |
| **3.legitimate.csv**            | Extracted features from legitimate URLs   | 5,000    | Processed from file 1                                        |
| **4.phishing.csv**              | Extracted features from phishing URLs     | 5,000    | Processed from file 2                                        |
| **5.urldata.csv**               | **Final dataset** - Combined feature data | 10,000   | Files 3 + 4                                                  |

## 🔄 Data Pipeline

```
Raw Data → Feature Extraction → Final Dataset
   ↓              ↓                 ↓
Files 1,2    →  Files 3,4     →   File 5
```

**Note:** Files 3-5 contain 17 engineered features used for machine learning model training.
