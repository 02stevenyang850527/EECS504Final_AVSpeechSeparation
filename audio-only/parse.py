import json

class JsonSerializable(object):

   def toDict(self):
      return self.__dict__

   def __repr__(self):
      return self.toDict()


class video(JsonSerializable):
   def __init__(self, yid, url, num_people):
      self.yid = yid
      self.url_list = [url]
      self.num_people_list = [num_people]

   def update(self, url, num_people):
      self.url_list += [url]
      self.num_people_list += [num_people]

   def __str__(self):
      return f"ID: {self.yid} has url_list: {url_list} and num_people_list {num_people_list}, respectively."

   def jsonable(self):
      return {"yid": self.yid, "urls": url_list, "peoples": num_people_list}


def write_json(filepath, data):
   with open(filepath, 'w', encoding='utf-8') as file_obj:
      json.dump(data, file_obj, ensure_ascii=False, indent=2)


if __name__ == '__main__':
   id_dict = {}

   with open('iden_split.txt', 'r') as f:
      for line in f:
         num, others = line.split(' ')
         yid, url, _ = others.split('/')
         if yid in id_dict.keys():
            id_dict[yid].update(url, num)
         else:
            id_dict[yid] = video(yid, url, num)

   for k in id_dict.keys():
      id_dict[k] = id_dict[k].toDict()

   write_json('training_list.json', id_dict)
