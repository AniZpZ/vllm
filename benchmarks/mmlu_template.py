import pandas as pd
import json
import os
from langchain.prompts import PromptTemplate
import openai
import random

template = PromptTemplate(
            input_variables=["question", "A", "B", "C", "D", "Answer"],
            template=
"""
USER: Question: {question}
A. {A}
B. {B}
C. {C}
D. {D} ASSISTANT: Answer: {Answer}
""",
        )

template_origin = PromptTemplate(
            input_variables=["question", "A", "B", "C", "D", "Answer"],
            template=
"""
Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer: {Answer}
""",
        )


template_with_analyse=PromptTemplate(
            input_variables=["question", "A", "B", "C", "D"],
            template=
"""
Q:{question}
(A) {A} (B) {B} (C) {C} (D) {D}
A: Let's think step by step.
""",
        )

class MMLUTemplate():

    def __init__(self, subject, file_path, is_analyse, origin):
        self.fiveShotTemplate = ""
        self.file_path = file_path
        self.subject = subject
        self.choices = ["A", "B", "C", "D"]
        self.is_analyse = is_analyse
        self.origin = origin
        if not is_analyse:
            if self.origin :
                self.getFewShotBaseTemplates_origin()
            else :
                self.getFewShotBaseTemplates()
        else:
            self.getFewShotBaseTemplateAnalyse()

    #########few_shot模板不带分析######################
    def getFewShotBaseTemplates(self, k=2):
        dev_df = pd.read_csv(self.file_path, header=None)

        self.fewShotTemplate = self.gen_prompt(dev_df,self.subject, k)
        return self.fewShotTemplate

    def gen_prompt(self,train_df, subject, k=1):
        prompt = "SYSTEM: The following are multiple choice questions (with answers) about {},Please select the correct answer from the options.".format(
            subject.replace('_',' '))


        for i in range(k):

            prompt += template.format(question=train_df.iloc[i, 0],A=train_df.iloc[i, 1],B=train_df.iloc[i, 2],C=train_df.iloc[i, 3],
                                      D=train_df.iloc[i, 4],Answer=train_df.iloc[i, 5] + "</s>")[1:-1]
        return prompt

    #########few_shot模板不带分析######################
    # when prompt is longer than the limit, use smaller k (e.g., k=2)
    def getFewShotBaseTemplates_origin(self, k=5):
        dev_df = pd.read_csv(self.file_path, header=None)

        self.fewShotTemplate = self.gen_prompt_origin(dev_df,self.subject, k)
        return self.fewShotTemplate

    def gen_prompt_origin(self,train_df, subject, k=1):
        prompt = "The following are multiple choice questions (with answers) about {},Please select the correct answer from the options.\n".format(
            subject.replace('_',' '))


        for i in range(k):

            prompt += template_origin.format(question=train_df.iloc[i, 0],A=train_df.iloc[i, 1],B=train_df.iloc[i, 2],C=train_df.iloc[i, 3],
                                      D=train_df.iloc[i, 4],Answer=train_df.iloc[i, 5] + "\n")[1:-1]
        return prompt




    ###############################################################3############




    ###########few_shot模板带分析，更改json文件就行###############################
    def getFewShotBaseTemplateAnalyse(self):
        mmlu_prompt = json.load(open('templates/lib_prompt/mmlu-cot.json'))
        self.fewShotTemplate=mmlu_prompt[self.subject]
        return self.fewShotTemplate



    #################获得模板############
    def getTemplate(self,test_df,i):
        if self.is_analyse==True:
            templ=template_with_analyse.format(question=test_df.iloc[i, 0],A=test_df.iloc[i, 1],B=test_df.iloc[i, 2],
                                         C=test_df.iloc[i, 3],D=test_df.iloc[i, 4])

            return self.fewShotTemplate + "\n" + templ

        else:
            if self.origin == True :
                    prompt_end = template_origin.format(question=test_df.iloc[i, 0],A=test_df.iloc[i, 1],B=test_df.iloc[i, 2],
                                                 C=test_df.iloc[i, 3],D=test_df.iloc[i, 4],Answer='')[1:-1]
            else:
                    prompt_end = template.format(question=test_df.iloc[i, 0],A=test_df.iloc[i, 1],B=test_df.iloc[i, 2],
                                         C=test_df.iloc[i, 3],D=test_df.iloc[i, 4],Answer='')[1:-1]
            return self.fewShotTemplate + prompt_end



################################################解析函数##################################################################
    def findAnswer(self,res):
        # print("模型输出为:", res)
        d = "NO"
        for d_ in res:
            if (ord(d_) >= 65 and ord(d_) <= 68):
                d = d_
                break
        # if d == 'NO' :
        #     d = random.choice(['A', 'B', 'C', 'D'])
        # print("答案解析为:", d)
        return d

    def findAnwerUsingRule(self,res):
        #print("模型输出为:", res)
        result = "NO"
        pattern = 'the answer is ('
        try:

            pred = res.lower().split(pattern)[1][0]

            if (ord(pred.upper()) >= 65 and ord(pred.upper()) <= 68):
                result = pred.upper()
        except:
            pass

        #print("答案解析为:",result)
        return result


if __name__ == "__main__":
    print(111)
    BT=mmlu_template("abstract_algebra",os.path.join('../Imsys/MMLU/data', "dev", "abstract_algebra" + "_dev.csv"),is_analyse=False)


    df=pd.read_csv(os.path.join('../Imsys/MMLU/data', "test", "abstract_algebra" + "_test.csv"), header=None)

    temp=BT.getTemplate(df,3)
    print('template==',temp)
