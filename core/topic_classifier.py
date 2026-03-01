import re
from typing import Dict, List, Tuple
from enum import Enum


class TopicType(Enum):
    JOB_HUNTING = "求职招聘"
    WORK_TALK = "工作讨论"
    STUDY = "学习交流"
    LIFE = "生活闲聊"
    NOTICE = "群公告"
    UNKNOWN = "其他"


class TopicClassifier:
    KEYWORDS = {
        TopicType.JOB_HUNTING: [
            "面试", "面经", "内推", "招聘", "实习", "校招", "社招", "HC", "简历",
            "offer", "薪资", "跳槽", "离职", "入职", "字节", "腾讯", "阿里",
            "百度", "美团", "滴滴", "京东", "华为", "小米", "蔚来", "理想",
            "算法岗", "开发岗", "产品岗", "运营岗", "测开", "前端", "后端",
            "java", "python", "go", "c++", "算法", "八股文", "手撕代码",
            "hr面", "技术面", "一面", "二面", "三面", "hr", "捞人"
        ],
        TopicType.WORK_TALK: [
            "工作", "项目", "需求", "bug", "线上", "加班", "周报", "日报",
            "开会", "评审", "架构", "设计", "优化", "性能", "压测", "上线",
            "回滚", "故障", "oncall", "值班", "绩效", "晋升", "答辩", "述职",
            "pua", "卷", "躺平", "摸鱼", "离职", "跳槽", "涨薪", "年终奖",
            "okr", "kpi", "敏捷", "迭代", "版本", "发布", "部署", "docker",
            "k8s", "微服务", "分布式", "高并发", "缓存", "数据库", "redis"
        ],
        TopicType.STUDY: [
            "课程", "作业", "考试", "论文", "毕设", "导师", "实验室", "科研",
            "论文", "期刊", "会议", "顶会", "cvpr", "icml", "nips", "aaai",
            "保研", "考研", "出国", "留学", "托福", "雅思", "gre", "gmat",
            "选课", "退课", "学分", "绩点", "gpa", "奖学金", "助教", "ta"
        ],
        TopicType.NOTICE: [
            "通知", "公告", "活动", "聚会", "聚餐", "团建", "报名", "截止",
            "@所有人", "所有人", "重要", "紧急", "请大家", "望周知"
        ]
    }
    
    RULES = [
        (re.compile(r'(内推|refer).*?(码|链接|邮箱)'), TopicType.JOB_HUNTING),
        (re.compile(r'(字节|腾讯|阿里|百度|美团).*?(面|招|聘)'), TopicType.JOB_HUNTING),
        (re.compile(r'\d+k|\d+万|年包|总包|薪资|待遇'), TopicType.JOB_HUNTING),
        (re.compile(r'(线上|生产|prod).*?(故障|bug|问题)'), TopicType.WORK_TALK),
        (re.compile(r'(加班|oncall|值班|熬夜).*?(到|至|了)'), TopicType.WORK_TALK),
        (re.compile(r'@所有人'), TopicType.NOTICE),
        (re.compile(r'^(通知|公告|【通知】|【公告】)'), TopicType.NOTICE),
    ]
    
    def __init__(self):
        self.keywords = self.KEYWORDS
        self.rules = self.RULES
    
    def classify(self, text: str) -> Tuple[TopicType, float]:
        text = text.lower()
        
        for pattern, topic in self.rules:
            if pattern.search(text):
                return topic, 0.9
        
        scores = {}
        for topic, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                count = text.count(keyword.lower())
                weight = len(keyword) / 10.0
                score += count * weight
            scores[topic] = score
        
        if not scores or max(scores.values()) == 0:
            return TopicType.LIFE, 0.5
        
        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]
        
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        return best_topic, min(confidence, 0.85)
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[TopicType, float]]:
        return [self.classify(text) for text in texts]
    
    def get_topic_stats(self, texts: List[str]) -> Dict[str, int]:
        stats = {topic.value: 0 for topic in TopicType}
        for text in texts:
            topic, _ = self.classify(text)
            stats[topic.value] += 1
        return stats


_classifier = None


def get_classifier() -> TopicClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TopicClassifier()
    return _classifier
