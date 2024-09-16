# Phân tích dữ liệu các bộ Anime, sau đó ứng dụng machine learning cơ bản để gợi Anime
## 1. Tổng quan về tập dữ liệu
### Anime.csv

| Thuộc tính   | Mô tả                                                                 |
|--------------|----------------------------------------------------------------------|
| **anime_id** | id duy nhất của myanimelist.net xác định một anime.                  |
| **name**     | Tên đầy đủ của anime.                                                |
| **genre**    | Danh sách các thể loại được phân tách bằng dấu phẩy của anime này.   |
| **type**     | Loại hình như phim, TV, OVA, v.v.                                    |
| **episodes** | Số tập trong chương trình này (1 nếu là phim).                       |
| **rating**   | Xếp hạng trung bình trên 10 của anime này.                           |
| **members**  | Số lượng thành viên cộng đồng tham gia vào "nhóm" của anime này.     |

## 2. Các công cụ xử dụng
Trong phân tích này, tôi đã sử dụng các thư viện và công cụ Python sau:

+ collections.Counter: Dùng để đếm tần suất xuất hiện của các phần tử trong một tập dữ liệu.
+ numpy: Hỗ trợ xử lý các phép toán số học và thao tác trên mảng dữ liệu lớn.
+ pandas: Thư viện mạnh mẽ để thao tác và phân tích dữ liệu dạng bảng (DataFrame).
+ matplotlib.pyplot: Sử dụng để vẽ biểu đồ và trực quan hóa dữ liệu.
+ seaborn: Công cụ hỗ trợ trực quan hóa dữ liệu, mở rộng từ matplotlib với các biểu đồ đẹp và dễ sử dụng hơn.
+ requests: Dùng để thực hiện các yêu cầu HTTP, giúp lấy dữ liệu từ web.
+ sklearn.feature_extraction.text.TfidfVectorizer: Dùng để chuyển đổi văn bản thành vector số (TF-IDF) để phân tích ngữ nghĩa và mức độ quan trọng của các từ trong văn bản.
+ sklearn.metrics.pairwise.sigmoid_kernel: Dùng để tính toán độ tương đồng giữa các vector văn bản.
+ matplotlib.ticker.MultipleLocator: Dùng để điều chỉnh thang đo và khoảng cách giữa các giá trị trên trục của biểu đồ.
## 3. Các bước tiền xử lý
### 3.1. Xử lý null
  Ở tập anime thì có các cột có gái trị null nhưu hình bên dưới
![image](https://github.com/user-attachments/assets/8f554ee9-369c-4582-82dc-423607721da9)

#### 3.1.a Xử lý null cột genre
def fill_genre(name):
    url = f'https://api.jikan.moe/v4/anime?q={name}&limit=1'
    for _ in range(5):
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data['data']:
                genres = data['data'][0].get('genres', [])
                return ', '.join(genre['name'] for genre in genres)
    return -1
Code này sẽ gọi API https://api.jikan.moe/v4/anime?q={name}&limit=1 với name là tên Anime, sau đó ta sẽ điền gái trị genre bị null vào 
#### 3.1.b Xử lý null cột type (Tương tự genre, vẫn dùng API và lấy type mình cần về)
#### 3.1.c Xử lý null cột Raying (Tương tự genre, vẫn dùng API và lấy rating anime mình cần về)
### 3.2. Xử lý trùng lặp: Xóa các hàng trùng lặp
## 4. Các phân tích đơn giản
+ Top anime có cộng đồng đông thành viên nhất
![image](https://github.com/user-attachments/assets/a27ed19b-31d6-46c4-9aaa-c47117868a37)

+ Các thể loại anime có thành viên là bao nhiêu và có rating trung bình là bao nhiêu
![image](https://github.com/user-attachments/assets/3a5552ea-8c48-4cd1-b0d8-3727c97ac8ab)

+ Phần trăm anime mỗi type
![image](https://github.com/user-attachments/assets/bd8777de-bb9f-4910-bf5f-262624665e92)

+ Trung bình rating và tổng thành viên mỗi type
  ![image](https://github.com/user-attachments/assets/cae4fa34-bab4-41aa-a046-887f655eea50)

+ các Anime rating cao nhất mỗi type
![image](https://github.com/user-attachments/assets/b891597a-eb50-4ff4-a6f0-55828e9bc160)

+ Các Anime có cộng đồng thành viên nhiều nhất mỗi type
![image](https://github.com/user-attachments/assets/75b92f8e-c8ac-4984-a073-59b008174663)


## 4. Hệ thống gợi ý đơn giản  ( Content Based Recommender )
from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)   

rec_indices = pd.Series(usedata.index, index = usedata["name"]).drop_duplicates()


# Recommendation Function
def give_recommendation(title, sig = sig):
    
    idx = rec_indices[title] # Getting index corresponding to original_title

    sig_score = list(enumerate(sig[idx]))  # Getting pairwsie similarity scores 
    sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
    sig_score = sig_score[1:11]
    anime_indices = [i[0] for i in sig_score]
     
    # Top 10 most similar movies
    rec_dic = {"No" : range(1,11), 
               "Anime Name" : usedata["name"].iloc[anime_indices].values,
               "Rating" : usedata["rating"].iloc[anime_indices].values}
    dataframe = pd.DataFrame(data = rec_dic)
    dataframe.set_index("No", inplace = True)
    
    print(f"Recommendations for {title} viewers :\n")
    
    return dataframe


![image](https://github.com/user-attachments/assets/c9286ad7-3b23-46b5-8f9d-97c23a8e63ce)

