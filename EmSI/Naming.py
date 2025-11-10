import os
import random
import openai
import pandas as pd
from openai import OpenAI

from SCDection.TableAnnotation import TableColumnAnnotation as TA
from learning.TFIDF import table_tfidf, roulette_wheel_selection


class GPTnaming:
    def __init__(self, apiKey,  format=0, sampling=None, sampling_number=5, header=False,
                 table_names=None):

        self._format = format
        self._sampling = sampling
        self._sampling_number = sampling_number
        self._header = header
        self._table_names = table_names
        self._message = []
        self._client = OpenAI(
            # This is the default and can be omitted
            api_key=apiKey
        )


    def sample(self, item):
        table = item.copy()
        if self._sampling is None:
            return table.head(self._sampling_number)
        if self._sampling == 'sample_cells':
            # sample half of the cells randomly
            sampleRowIdx = []
            for _ in range(self._sampling_number):
                sampleRowIdx.append(random.randint(0, len(table) - 1))
            for ind in sampleRowIdx:
                for col_idx in range(len(item.columns) - 1):
                    table.iloc[ind, col_idx] = ""
        elif self._sampling == 'sample_cells_TFIDF':
            table = table.astype(str)
            df_tfidf = table_tfidf(table)
            augmented_cols = []
            for col in table.columns:
                selected_indices = roulette_wheel_selection(df_tfidf[col].index, self._sampling_number,
                                                            df_tfidf[col])
                augmented_col = table[col].iloc[selected_indices].reset_index(drop=True)
                augmented_cols.append(augmented_col)
                # Combine the augmented columns to form the new table
            table = pd.concat(augmented_cols, axis=1)
        return table

    def prompt_clusters(self, cluster):

        """
        :param clusters: the input cluster of items for naming
            0. column: each of the item is a column, we take column as a format
            1. text: each of the item is a column, we transform the column into text format
            2. table: each of the item is a dataframe
            3. table: each of the item is a dataframe and mark the subject attribute
            4. table: each of the item is a dataframe but we only use subject attributes
        """
        prompt = ""
        for index, item in enumerate(cluster):
            data_item = self.sample(item)
            if self._format == 0:
                column_text = data_item.to_string(index=False, header=False)
                prompt += f"Column {index}: " + column_text
                prompt += f". Header:{data_item.header};" if self._header is not None else "."
            elif self._format == 1:
                print("Currently not supported!")
                return
            elif self._format == 2:
                    prompt += f"<table> {index}: \n" + "||".join(map(str, data_item.columns)) + "\n"
                    for index, row in data_item.iterrows():
                        prompt += "||".join(map(str, row.values)) + "\n"
                    prompt += "</table>"
            elif self._format >2:
                    subcol_index = -1
                    annotation_table = TA(data_item)
                    annotation_table.subcol_Tjs()
                    NE_column_score = annotation_table.column_score

                    if len(NE_column_score) > 0:
                        subcol_index = [key for key, value in NE_column_score.items() if
                                        value == max(NE_column_score.values())][0]
                    if subcol_index == -1:
                        prompt += f"Table {index}: \n" +"||".join(map(str, data_item.columns)) + "\n"
                        for index, row in data_item.iterrows():
                            prompt += "||".join(map(str, row.values)) + "\n"
                    else:
                        if self._format == 3:
                            prompt += (f"Table {index}: \n" +
                                       "||".join(
                                f"<sc>{col}</sc>" if i == subcol_index else col for i, col in
                                enumerate(data_item.columns)) + "\n")
                            for index, row in data_item.iterrows():
                                prompt += "||".join(f"<sc>{value}</sc>" if i == subcol_index else str(value) for i, value in
                                                    enumerate(row.values)) + "\n"

                        if self._format == 4:
                            column = list(data_item.columns)[subcol_index]
                            column_text = ",".join(map(str, data_item[column].tolist()))
                            #prompt += f"Subject attribute of table {index}: " + column_text
                            prompt +=  f"<sc> {column}: " if self._header is True else  f"<sc> "
                            prompt += column_text +" </sc>"
                            #prompt += f". Header: {column};" if self._header is True else "."
            if self._table_names is not None:
                    prompt += f"table name: {self._table_names[index]}."
            prompt += "\n"
        if len(prompt) > 10000:
            prompt = prompt[:10000]
        return prompt

    def generate_answers(self, cluster, task, reply=None, instructions: list = None, shorts: tuple = None, AI_instructions: list = None,
                         newMessage=False):

        if newMessage is True:
            self._message.clear()
        prompt =   self.prompt_clusters(cluster)
        new_assistant_reply = ""
        if instructions is None:
            self._message.append({"role": "user", "content": task + "\n" + prompt})
        else:
            if reply is not None:
                self._message.append({"role": "system", "content": reply})
            for instruction in instructions:
                self._message.append({"role": "system", "content": instruction})
            if shorts is not None:
                self._message.append({"role": "user", "content": shorts[0]})
                self._message.append({"role": "assistant", "content": shorts[1]})
            if AI_instructions is not None:
                for instruction in AI_instructions:
                    self._message.append({"role": "assistant", "content": instruction})
            self._message.append({"role": "user", "content": task + "\n" + prompt})

        try:
            #print(self._message)
            response = self._client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=self._message,
                )
            new_assistant_reply = response.choices[0].message.content
        except Exception as e:
                print(f"An error occurred: {e}")
        return new_assistant_reply

class TableType:
    def __init__(self, apiKey,table:pd.DataFrame):
        self._table = table.head(10) #TODO this needs refine
        self._specific_Type = []
        self._messages = []
        self._client = OpenAI(
            # This is the default and can be omitted
            api_key=apiKey
        )
    def table_type(self, entity_typeT = None):
        table_prompt =  "||".join(map(str, self._table.columns)) + "\n"
        for index, row in self._table.iterrows():
                        table_prompt += "||".join(map(str, row.values)) + "\n"
        topLeveltype_Ins = ""
        if entity_typeT is not None:
            topLeveltype_Ins = f"while the the inferred mutual entity type of the table's belonging cluster is {entity_typeT},"
        self._messages.append({"role": "user",
                               "content": f"Given the table, {topLeveltype_Ins} please identify the conceptual entity type that best describes the table. The table is {table_prompt}."})
        self._messages.append({"role": "system",
                               "content": "Output will ONLY BE the most relevant candidate name for the identified specific conceptual entity type of this table, NO OTHER TEXT!"})
        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self._messages,
            )
            specific_type = response.choices[0].message.content
           # self._specific_Type.append(specific_type)
            return specific_type
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""
    def judge_ancestor(self, identified_type, entity_typeT):
        self._messages.append({"role": "user","content": f"The identified conceptual type of this table's cluster is {entity_typeT}. "
                                                        f"Please judge if the specific type {identified_type} of this table is a sub-concept of the cluster's identified type {entity_typeT}."})
        self._messages.append({"role": "system",
                               "content": "Only output: True/False"})
        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self._messages,
            )
            judge = response.choices[0].message.content
            return judge
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""
