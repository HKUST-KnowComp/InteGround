""" Adapted from the original prompt used in the paper. We modified the format to make it easier to parse the results. """

import sys
sys.path.append("..")

from utils import get_dict
from nltk.tokenize import sent_tokenize
from mykits import get_llm_response

factscore_decompose_prompt_full = """Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun’s Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.
- He made his acting debut in the film.
- He made his acting debut in The Moon is the Sun’s Dream.
- The Moon is the Sun’s Dream is a film.
- The Moon is the Sun’s Dream was released in 1992.
- After his acting debut, he appeared in small and supporting roles.
- After his acting debut, he appeared in small and supporting roles throughout the 1990s.

Please breakdown the following sentence into independent facts: He is also a successful producer and engineer, having worked with a wide variety of artists,
including Willie Nelson, Tim McGraw, and Taylor Swift.
- He is successful.
- He is a producer.
- He is a engineer.
- He has worked with a wide variety of artists.
- Willie Nelson is an artist.
- He has worked with Willie Nelson.
- Tim McGraw is an artist.
- He has worked with Tim McGraw.
- Taylor Swift is an artist.
- He has worked with Taylor Swift.

Please breakdown the following sentence into independent facts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served
as the back-up Command Module Pilot for the Gemini 7 mission.
- Collins became an astronaut.
- Collins became one of the third group of astronauts.
- Collins became one of the third group of astronauts selected.
- Collins became one of the third group of astronauts selected by NASA.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- He served as the Command Module Pilot.
- He served as the back-up Command Module Pilot.
- He served as the Command Module Pilot for the Gemini 7 mission.

Please breakdown the following sentence into independent facts: In addition to his acting roles, Bateman has written and directed two short films and is
currently in development on his feature debut.
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman has written and directed two short films.
- Bateman is currently in development on his feature debut.

Please breakdown the following sentence into independent facts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who
was the Command Module Pilot for the Apollo 11 mission in 1969.
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Please breakdown the following sentence into independent facts: He was an American composer, conductor, and musical director.
- He was an American.
- He was a composer.
- He was a conductor.
- He was a musical director.

Please breakdown the following sentence into independent facts: She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.
- She currently stars in Love and Destiny.
- Love and Destiny is a romantic comedy series.
- Love and Destiny premiered in 2019.

Please breakdown the following sentence into independent facts: During his professional career, McCoy played for the Broncos, the San Diego Chargers, the
Minnesota Vikings, and the Jacksonville Jaguars.
- McCoy played for the Broncos.
- McCoy played for the Broncos during his professional career.
- McCoy played for the San Diego Chargers.
- McCoy played for the San Diego Chargers during his professional career.
- McCoy played for the Minnesota Vikings.
- McCoy played for the Minnesota Vikings during his professional career.
- McCoy played for the Jacksonville Jaguars.
- McCoy played for the Jacksonville Jaguars during his professional career.

Please breakdown the following sentence into independent facts: {}\n"""


factscore_decompose_prompt = """Example 0:
Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun’s Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.
{{"facts": ["He made his acting debut in the film.", "He made his acting debut in The Moon is the Sun\\u2019s Dream.", "The Moon is the Sun\\u2019s Dream is a film.", "The Moon is the Sun\\u2019s Dream was released in 1992.", "After his acting debut, he appeared in small and supporting roles.", "After his acting debut, he appeared in small and supporting roles throughout the 1990s."]}}

Example 1:
Please breakdown the following sentence into independent facts: He is also a successful producer and engineer, having worked with a wide variety of artists,
including Willie Nelson, Tim McGraw, and Taylor Swift.
{{"facts": ["He is successful.", "He is a producer.", "He is a engineer.", "He has worked with a wide variety of artists.", "Willie Nelson is an artist.", "He has worked with Willie Nelson.", "Tim McGraw is an artist.", "He has worked with Tim McGraw.", "Taylor Swift is an artist.", "He has worked with Taylor Swift."]}}

Example 2:
Please breakdown the following sentence into independent facts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served
as the back-up Command Module Pilot for the Gemini 7 mission.
{{"facts": ["Collins became an astronaut.", "Collins became one of the third group of astronauts.", "Collins became one of the third group of astronauts selected.", "Collins became one of the third group of astronauts selected by NASA.", "Collins became one of the third group of astronauts selected by NASA in 1963.", "He served as the Command Module Pilot.", "He served as the back-up Command Module Pilot.", "He served as the Command Module Pilot for the Gemini 7 mission."]}}

Example 3:
Please breakdown the following sentence into independent facts: {}\n"""


def factscore_decompose(input_text, model="claude-v3-sonnet"):
    """ Implemented according to the original implementation at https://github.com/shmsw25/FActScore/blob/main/factscore/atomic_facts.py
    We adapt the official prompts to reduce format errors during parsing.
    """
    # Get a list of sentences.
    sentences = sent_tokenize(input_text)

    # Obtain factscore decompositions for each sentence in the input text.
    all_factoids = []
    for sent in sentences:
        retry = 0
        while retry < 3:
            try:
                res = get_llm_response(factscore_decompose_prompt.format(sent), model=model)
                # res = res[res.index("-"):]
                res = eval(get_dict(res))['facts']
                # res = [item.strip("- ") for item in res.strip().split('\n') if item]
                all_factoids.extend(res)
                break
            except Exception as e:
                print("[In factscore decomposition]", e, retry)
                retry += 1

    return all_factoids