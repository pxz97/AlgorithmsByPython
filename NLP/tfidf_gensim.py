import jieba
from gensim import corpora, models
from gensim.similarities.docsim import MatrixSimilarity

# 0. 生成数据
base_data = [
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。野径云俱黑，江船火独明。晓看红湿处，花重锦官城。",
    "君问归期未有期，巴山夜雨涨秋池。何当共剪西窗烛，却话巴山夜雨时。",
    "莫听穿林打叶声，何妨吟啸且徐行。竹杖芒鞋轻胜马，谁怕？一蓑烟雨任平生。料峭春风吹酒醒，微冷，山头斜照却相迎。回首向来萧瑟处，归去，也无风雨也无晴。",
    "天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。",
    "古木阴中系短篷，杖藜扶我过桥东。沾衣欲湿杏花雨，吹面不寒杨柳风。",
    "少年听雨歌楼上。红烛昏罗帐。壮年听雨客舟中。江阔云低、断雁叫西风。 而今听雨僧庐下。鬓已星星也。悲欢离合总无情。一任阶前、点滴到天明。",
    "雨里鸡鸣一两家，竹溪村路板桥斜。妇姑相唤浴蚕去，闲看中庭栀子花。",
    "一夕轻雷落万丝，霁光浮瓦碧参差。有情芍药含春泪，无力蔷薇卧晓枝。"
]

# 1. 将base_data中的8个诗句遍历进行分词
base_items = [[i for i in jieba.cut(item)] for item in base_data]
count = 0
for item in base_items:
    count += len(item)

dictionary = corpora.Dictionary(base_items)

corpus = [dictionary.doc2bow(item) for item in base_items]

tfidf = models.TfidfModel(corpus)

num_features = len(dictionary.token2id.keys())

index = MatrixSimilarity(tfidf[corpus], num_features=num_features)

test_text = "风雨凄凄，鸡鸣喈喈。既见君子，云胡不夷。风雨潇潇，鸡鸣胶胶。既见君子，云胡不瘳。风雨如晦，鸡鸣不已。既见君子，云胡不喜。"
test_words = [word for word in jieba.cut(test_text)]

new_vec = dictionary.doc2bow(test_words)

sims = index[tfidf[new_vec]]
