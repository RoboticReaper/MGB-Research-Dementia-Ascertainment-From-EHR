from tokenizer import tokenize
import re
from temporal_lexicons import *

class Temporal():

    def __init__(self):

        # set up lexicons / dictionaries

        weekdays = "monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|Mon\.|Tue\.|Wed\.|Thur\.|Fri\.|Sat\.|Sun\."

        week_hash = {
            'Mon.': 'monday',
            'Tue.': 'tuesday',
            'Wed.': 'wednesday',
            'Thur.': 'thursday',
            'Fri.': 'friday',
            'Sat.': 'saturday',
            'Sun.': 'sunday'
        }

        vagueness = "just about|about|approximately|around|roughly|something like|more or less|in the region of|generally"

        uncertainty = "just about|about|approximately|around|roughly|something like|more or less|in the region of|generally"

        am_pm1 = "am|pm|a\.m\.|p\.m\."
        am_pm2 = "in the morning|in the afternoon|in the evening|at night"

        pod = "morning|mid-day|afternoon|evening|night|midnight|daytime|am|pm"

        minute_quan = "five|ten|fifteen|quarter|half"

        time_det = "his|her|the patient's|the|this|current|present"

        pdir = "more than|greater than|less than|almost|nearly|no less than|at least|no more than|at most|a total of|full"

        self.before_pdir_hash = {
            'more than': 'before',
            'greater than': 'before',
            'less than': 'after',
            'nearly': 'after',
            'almost': 'after',
            'no less than': 'equal_or_before',
            'at least': 'equal_or_before',
            'no more than': 'equal_or_after',
            'at most': 'equal_or_after',
            'a total of': 'equal',
            'full': 'equal'
        }
        
        self.after_pdir_hash = {
            'more than': 'after',
            'greater than': 'after',
            'less than': 'before',
            'nearly': 'before',
            'almost': 'before',
            'no less than': 'equal_or_after',
            'at least': 'equal_or_after',
            'no more than': 'equal_or_before',
            'at most': 'equal_or_before',
            'a total of': 'equal',
            'full': 'equal'
        }
        
        self.dur_pdir_hash = {
            'more than': 'before',
            'greater than': 'before',
            'less than': 'after',
            'nearly': 'after',
            'almost': 'after',
            'no less than': 'equal_or_before',
            'at least': 'equal_or_before',
            'no more than': 'equal_or_after',
            'at most': 'equal_or_after',
            'a total of': 'equal',
            'full': 'equal'
        }
        
        self.pdir_hash = {
            'more than': 'more_than',
            'greater than': 'more_than',
            'less than': 'less_than',
            'nearly': 'less_than',
            'almost': 'less_than',
            'no less than': 'equal_or_more',
            'at least': 'equal_or_more',
            'no more than': 'equal_or_less',
            'at most': 'equal_or_less',
            'a total of': 'equal',
            'full': 'equal'
        }

        before_prep1 = "before|prior to|previous to|previously|earlier than|sooner than|ahead of|facing|in front of|to|toward"
        before_prep2 = "before|prior to|previous to|previously|earlier than|sooner than|ahead of|facing|in front of"

        eq_before1 = "by|through"
        eq_before2 = "by"

        after_prep = "after|following|next to|thereafter|post|subsequent to|behind|later than|once"

        eq_after = "since|from"
        
        simul_prep1 = "at|on|in|over|throughout|simultaneous|simultaneously|of|and"
        simul_prep2 = "at|on|in|over|throughout|simultaneous|simultaneously"
        simul_prep3 = "at|on|in|simultaneous|simultaneously|and|between"
        simul_prep4 = "at|on|in|simultaneous|simultaneously|of|between"

        within1 = "during|within"
        within2 = "in|within|during"
       
        as_of1 = "for|lasting|last|over|all through|through|times|time|throughout"
        as_of2 = "for|lasting|over|all through|throughout"
        as_of3 = "for|lasting|over|all through|through|throughout"
        as_of4 = "lasting|over|all through|through|throughout"
        as_of5 = "for|lasting|last|over|all through|throughout|through|times|time|a history of a"

        duration_word = as_of5 + "|" + within2
        
        # used for time
        direction_word1 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before1 + "|" + simul_prep1 + "|" + within1 + "|until"
        direction_word2 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before1 + "|" + simul_prep2 + "|" + within1 + "|until"
        # used for duration
        direction_word4 = after_prep + "|" + eq_after + "|" + before_prep2 + "|" + eq_before1 + "|" + simul_prep2 + "|" + within1 + "|until"
        direction_word7 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before2 + "|" + simul_prep3 + "|" + within1 + as_of3 + "|until"
        # used for reference event
        direction_word3 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before1 + "|" + simul_prep3 + "|" + within1 + as_of1 + "|until"
        direction_word5 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before1 + "|" + simul_prep3 + "|" + within1 + "|until"
        direction_word8 = after_prep + "|" + eq_after + "|" + before_prep2 + "|" + eq_before2 + "|" + simul_prep4 + "|" + within1 + as_of4 + "|until"
        # used used for date
        direction_word6 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before1 + "|" + simul_prep3 + "|" + within1 + as_of2 + "|until"
        direction_word9 = after_prep + "|" + eq_after + "|" + before_prep1 + "|" + eq_before1 + "|" + simul_prep1 + "|" + within1 + as_of2 + "|until"

        start_part1 = "the beginning of|early|the start of|the dawn of|earlier|dawn"
        end_part1 = "the end of|late|later"
        mid_part1 = "the middle of|mid-|mid|middle"
        part1 = start_part1 + "|" + mid_part1 + "|" + end_part1

        start_part2 = "early|dawn|earlier"
        end_part2 = "later"
        mid_part2 = "middle"
        part2 = start_part2 + "|" + mid_part2 + "|" + end_part2

        rest_part = "the rest of|the remainder of"

        quantity_word = "a|an|[0-9]+?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninty"
        quantity_head = "twenty[ |-]|thirty[ |-]|forty[ |-]|fifty[ |-]|sixty[ |-]|seventy[ |-]|eighty[ |-]|ninty[ |-]"
        quantity_tail = "[ |-]hundred|[ |-]thousand|[ |-]million|[ |-]billion"
        
        ord_num_word = "first|second|third|[0-9]+?th|[0-9]+?st|[0-9]+?nd|[0-9]+?rd|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|ninteenth|twentieth"

        self.quantity_hash = {
            'a': '1',
            'an': '1',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
            'eleven': '11',
            'twelve': '12',
            'thirteen': '13',
            'fourteen': '14',
            'fifteen': '15',
            'sixteen': '16',
            'seventeen': '17',
            'eighteen': '18',
            'ninteen': '19',
            'twen': '20',
            'thir': '30',
            'fort': '40',
            'fift': '50',
            'sixt': '60',
            'seve': '70',
            'eigh': '80',
            'nint': '90',
            'hundred': '100',
            'thousand': '1000',
            'million': '1000000',
            'billion': '1000000000'
        }

        self.ord_num_hash = {
            'first': '1',
            'second': '2',
            'third': '3',
            'fourth': '4',
            'fifth': '5',
            'sixth': '6',
            'seventh': '7',
            'eighth': '8',
            'ninth': '9',
            'tenth': '10',
            'eleventh': '11',
            'twelfth': '12',
            'thirteenth': '13',
            'fourteenth': '14',
            'fifteenth': '15',
            'sixteenth': '16',
            'seventeenth': '17',
            'eighteenth': '18',
            'ninteenth': '19',
            'twentieth': '20'
        }

        tunit_word3 = "minute|hour|day|week|night|morning|afternoon"
        tunit_word4 = "second|minute|hour|day|week|month|year|night|morning|afternoon"
        tunit_word5 = "second|minute|hour|day|week|night|morning|afternoon"

        seasons = "spring|springtime|summer|summertime|fall|autumn|fall semester|fall term|winter|wintertime"

        old = " old|\-old"

        course = "course|of|course of"

        det_word1 = "his|her|the patient's|the date of|the time of|time of|the day of|the period of|the present time|the"
        det_word2 = "his|her|the patient's|the|this|current|present"
        det_word3 = "his|her|the patient's|this|the"
        det_word4 = "his|her|the patient's|this|that|those|these|the|a period of"

        det_word_event = "his|her|the patient's|the date of|the time of|time of|the day of|the perod of|the present time|the duration of|the entire|the course of|the whole|the"
        
        month_word1 = "January|january|February|february|March|April|april|May|may|June|june|July|july|August|august|September|september|October|october|November|november|December|december|Jan\.|jan\.|Feb\.|feb\.|Mar\.|mar\.|Apr\.|apr\.|Jun\.|jun\.|Jul\.|Aug\.|aug\.|Sept\.|sept\.|Sep\.|sep\.|Oct\.|oct\.|Nov\.|nov\.|Dec\.|dec\."
        month_word2 = "January|january|February|february|March|April|april|May|may|June|june|July|july|August|august|September|september|October|october|November|november|December|december"
        
        decade_word = "twenties|thirties|forties|fifties|sixties|seventies|eighties|ninties"

        self.decade_hash = {
            'twen': '2',
            'thir': '3',
            'fort': '4',
            'fift': '5',
            'sixt': '6',
            'seve': '7',
            'eigh': '8',
            'nint': '9'
        }

        adj_num1 = "several|a few|a couple of|few"
        adj_num2 = "many|some|a number of|multiple"
        adj_num3 = "some period of time|a period of time"

        self.adj_num_hash = {
            'several': '3-9',
            'a few': '2-10',
            'a couple of': '2-7',
            'few': '2-10'
        }

        before_adj = "last|past|preceding|prior|previously|previous|most recent"

        after_adj = "the next|next|coming|following|the following|subsequent|future|ensuing"

        simul_adj = "this|current|present|the|these"

        adj_word1 = before_adj + "|" + after_adj + "|" + simul_adj
        adj_word2 = before_adj + "|" + after_adj

        #self.admit_date_rule_list = []

        #self.admit_date_rule_list.append(re.compile('date\sof\svisit\s*\:\s*(0[1-9]|[1-9]|10|11|12)\s*[\/\-]\s*([0-2][1-9]|[1-9]|30|31)\s*[\/\-]\s*(\d{2}|\d{4})'))
        #self.admit_date_rule_list.append(re.compile('admission\s*\:\s*(0[1-9]|[1-9]|10|11|12)\s*[\/\-]\s*([0-2][1-9]|[1-9]|30|31)\s*[\/\-]\s*(\d{2}|\d{4})'))
        #self.admit_date_rule_list.append(re.compile('\|+(0[1-9]|[1-9]|10|11|12)\s*[\/\-]\s*([0-2][1-9]|[1-9]|30|31)\s*[\/\-]\s*(\d{2}|\d{4})\s{1,3}(\d{2})\s*\:\s*(\d{2})\|+'))

        # initialize lists to hold the rules

        self.time_rule_list = []
        self.date_rule_list = []
        self.duration_rule_list = []

        # create time, date, and duration rules and add them to their respective rule lists

        # TODO: come up with more informative rule names?

        """
        TIME RULES

        Name: Time0
        Examples: "at about 8:00 in the evening"

        Name: Time1
        Examples: "before 8 o'clock at night"

        Name: Time2
        Examples: "at about quarter past eight in the evening"

        Name: Time3
        Examples: "late at night"

        """

        self.time_rule_list.append(re.compile("(" + direction_word1 + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(\d{1,2})\s*(:)\s*(\d{2})(\s{1,3})?(" + am_pm1 +")?(/s+)?(" + am_pm2 + ")?"))
        self.time_rule_list.append(re.compile("(" + direction_word1 + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(" + quantity_word + ")(\s{1,3})(o\s\'\sclock|" + am_pm2 + ")(\s{1,3})?(" + am_pm2 + ")?"))
        self.time_rule_list.append(re.compile("(" + direction_word1 + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(" + quantity_word + ")?(" + minute_quan + ")(\s{1,3})(minutes)?(\s{1,3})?(past|to)(\s{1,3})(" + quantity_word + ")(\s{1,3})?(o\'clock)?(\s{1,3})?(" + am_pm2 + ")?"))
        self.time_rule_list.append(re.compile("(" + part1 + ")?(\s{1,3})?(" + direction_word1 + ")(\s{1,3})(" + time_det + ")?(\s{1,3})?(" + adj_word1 + ")?(\s{1,3})?(" + pod + ")"))

        """
        DATE RULES

        Name: Date0
        Examples: "on 06/21/16"

        Name: Date1
        Examples: "before 04-2013"

        Name: Date2
        Examples: "until 05-22"

        Name: Date3
        Examples: "on may 22, 1992"

        Name: Date4
        Examples: "in may of 1992"

        Name: Date5
        Examples: "prior to april 13th"

        Name: Date6
        Examples: "throughout early august"

        Name: Date7
        Examples: "before the 25th of december"

        Name: Date8
        Examples: "prior to late last year"

        Name: Date9
        Examples: "from about mid 2010"

        Name: Date10
        Examples: "for roughly last year"

        Name: Date11
        Examples: "in the late 1990's"

        Name: Date12
        Examples: 

        """

        self.date_rule_list.append(re.compile("(" + direction_word6 + ")?(\s{1,3})?(01|02|03|04|05|06|07|08|09|2|3|4|5|6|7|8|9|10|11|12|1)\s*(\/|\-)\s*(01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|1|2|3|4|5|6|7|8|9)\s*(\/|\-)\s*(19[0-9]{2}|20[0-4][0-9]|[0-9]{2})"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")?(\s{1,3})?(01|02|03|04|05|06|07|08|09|2|3|4|5|6|7|8|9|10|11|12|1)\s*(\/|\-)\s*(19[0-9]{2}|20[0-4][0-9]|[3][2-9]|[4-9][0-9]|0[0-9])"))
        self.date_rule_list.append(re.compile("(" + direction_word9 + ')(\s{1,3})(01|02|03|04|05|06|07|08|09|10|11|12|1|2|3|4|5|6|7|8|9)(\s{1,3})?(\/|\-)(\s{1,3})?(0[1-9]|[1-2][0-9]|3[0-1]|[1-9])'))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")?(\s{1,3})?(" + month_word1 + ")(\s{1,3})(" + ord_num_word + "|[1-2][0-9]|3[0-1]|[1-9])(\s{1,3}|\,\s*)(19[0-9]{2}|20[0-4][0-9])"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")?(\s{1,3})?(" + month_word1 + ")(\s{1,3}|\,\s{1,3}|\s{1,3}of\s{1,3})(19[0-9]{2}|20[0-4][0-9])"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")(\s{1,3})(" + month_word1 + ")(\s{1,3})(" + ord_num_word + "|[1-2][0-9]|3[0-1]|[1-9])"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")(\s{1,3})(" + part1+ ")?(\s{1,3})?(" + month_word1 + ")"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")?(\s{1,3})(the)(\s{1,3})(" + ord_num_word + ")(\s{1,3})(of)(\s{1,3})(" + month_word1 + ")"))
        self.date_rule_list.append(re.compile("\s(" + direction_word6 + ")?(\s)?(" + part1 + ")(\s{1,3})(" + adj_word1 + ")?(\s{1,3})(" + month_word1 + "|month|year)"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(" + part1+ ")?(\s{1,3})?(19[0-9]{2}|20[0-4][0-9])"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")?(\s{1,3})?(" + vagueness + ")?(\s{1,3})?(" + det_word3 + ")?(\s{1,3})?(" + adj_word1 + ")(\s{1,3})(" + month_word1 + "|month|year)"))
        self.date_rule_list.append(re.compile("(" + direction_word6 + ")(\s{1,3})(" + det_word3 + ")?(\s{1,3})(" + part1+ ")?(\s{1,3})?(19\d0\'?s|" + decade_word + ")"))
        self.date_rule_list.append(re.compile("(" + simul_prep1 + "|from|" + as_of3 + ")?(\s{1,3})?(" + month_word1 + ")(\s{1,3})(" + ord_num_word + "|[1-2][0-9]|3[0-1]|[1-9])(\s{1,3})?(to|\-|and)(\s{1,3})?(the)?(\s{1,3})?(" + ord_num_word + "|[1-2][0-9]|3[0-1]|[1-9])"))

        """
        DATE RULES

        Name: Duration0
        Examples: 

        Name: Duration1
        Examples: 

        Name: Duration2
        Examples: 

        Name: Duration3
        Examples: 

        Name: Duration4
        Examples: 

        Name: Duration5
        Examples: 

        Name: Duration6
        Examples: 

        Name: Duration7
        Examples: 

        Name: Duration8
        Examples: 

        Name: Duration9
        Examples: 

        Name: Duration10
        Examples: 

        Name: Duration11
        Examples: 

        Name: Duration12
        Examples: 

        Name: Duration13
        Examples: 

        """

        self.duration_rule_list.append(re.compile("(" + duration_word + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(" + det_word_event+ ")?(\s{1,3})?(" + adj_word2 + ")?(\s{1,3})?(\d{1,4})(\s{1,3})?(\-)(\s{1,3})?(\d{1,4})(\s{1,3})(" + tunit_word4 + ")(s)?(?!old)"))
        self.duration_rule_list.append(re.compile("(" + duration_word + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(" + det_word_event+ ")?(\s{1,3})?(" + adj_word2 + ")?(\s{1,3})?(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})?(" + tunit_word4 + ")?(s)?(\s{1,3})(to|ot|or)(\s{1,3})(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})(" + tunit_word4 + ")(s)?"))
        self.duration_rule_list.append(re.compile("(" + duration_word + ")(\s{1,3})(" + det_word_event + ")?(\s{1,3})?(" + adj_word2 + ")?(\s{1,3})?(" + adj_num1 + "|" + adj_num2 + ")(\s{1,3})(" + tunit_word4 + ")(s)?"))
        self.duration_rule_list.append(re.compile("(" + duration_word + ")(\s{1,3})(" + vagueness + ")?(\s{1,3})?(" + pdir + ")?(\s{1,3})?(" + det_word_event + ")?(\s{1,3})?(" + adj_word2 + ")?(\s{1,3})?(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})(" + tunit_word4 + ")(s)?"))
        self.duration_rule_list.append(re.compile("(" + duration_word + ")(\s{1,3})(" + det_word_event + ")?(\s{1,3})?(" + adj_word2 + ")?(\s{1,3})?(" + tunit_word4 + ")(s)"))
        self.duration_rule_list.append(re.compile("(with|has|had|of|to complete|to finish|received|finished|receiving)(\s{1,3})(a\s{1,3})?(\d{1,4})(\s{1,3})?(\-)(\s{1,3})?(\d{1,4})(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(long\s{1,3})?(history|duration|" + course + ")(\s{1,3}of)?"))
        self.duration_rule_list.append(re.compile("(with|has|had|of|to complete|to finish|received|finished|receiving)(\s{1,3})(a\s{1,3})?(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})?(" + tunit_word4 + ")?(s)?(\s{1,3})(to|ot|or)(\s{1,3})(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(long\s{1,3})?(history|duration|" + course + ")(\s{1,3}of)?"))
        self.duration_rule_list.append(re.compile("(with|has|had|of|to complete|to finish|received|finished|receiving)(\s{1,3})(a\s{1,3})?(" + adj_num1 + "|" + adj_num2 + ")(\s{1,3})?(\-)?(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(long\s{1,3})?(history|duration|" + course + ")(\s{1,3}of)?"))
        self.duration_rule_list.append(re.compile("(with|has|had|of|to complete|to finish|received|finished|receiving)(\s{1,3})(a\s{1,3})?(" + vagueness + ")?(\s{1,3})?(" + pdir + ")?(\s{1,3})?(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})?(\-)?(\s{1,3})?(" + tunit_word4 + ")(s)?(\s{1,3})(long\s{1,3})?(history|duration|" + course + ")(\s{1,3}of)"))
        self.duration_rule_list.append(re.compile("(" + vagueness + ")?(\s{1,3})?(" + det_word_event+ ")?(\s{1,3})?(" + adj_word1 + ")?(\s{1,3})?(\d{1,4})(\s{1,3})?(\-)(\s{1,3})?(\d{1,4})(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(" + before_prep2 + "|" + after_prep + ")(\.|\!|\:|\;|\,|\s{1,3})"))
        self.duration_rule_list.append(re.compile("(" + vagueness + ")?(\s{1,3})?(" + det_word_event+ ")?(\s{1,3})?(" + adj_word1 + ")?(\s{1,3})?(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})?(" + tunit_word4 + ")(s)?(\s{1,3})(to|ot|or)(\s{1,3})(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(" + before_prep2 + "|" + after_prep + ")(\.|\!|\:|\;|\,|\s{1,3})"))
        self.duration_rule_list.append(re.compile("(" + det_word_event+ ")?(\s{1,3})?(" + adj_word1 + ")?(\s{1,3})?(" + adj_num1 + "|" + adj_num2 + ")(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(" + before_prep2 + "|" + after_prep + ")(\.|\!|\:|\;|\,|\s{1,3})"))
        self.duration_rule_list.append(re.compile("(" + vagueness + ")?(\s{1,3})?(" + pdir + ")?(\s{1,3})?(" + det_word_event+ ")?(\s{1,3})?(" + adj_word1 + ")?(\s{1,3})?(" + quantity_head + ")?(" + quantity_word + ")(" + quantity_tail + ")?(\s{1,3})(" + tunit_word4 + ")(s)?(\s{1,3})(" + before_prep2 + "|" + after_prep + ")(\.|\!|\:|\;|\,|\s{1,3})"))
        self.duration_rule_list.append(re.compile("(" + part2 + ")?(\s{1,3})?(" + direction_word1 + ")?(\s{1,3})?(" + det_word_event+ ")?(\s{1,3})(" + adj_word1 + ")(\s{1,3})(" + tunit_word3 + ")(s)?"))

    def proc_time0(self, direction, sp1, vagueword, sp2, h, colon, m, sp3, apm1, sp4, apm2):
        output = {
            'event_point': '',
            'anchor': '',
            'anchor_mod': '',
            'relation': trans_dir(direction),
            'vagueness': '',
            'rule': 'Time0'
        }
        if vagueword:
            output['vagueness'] = 'yes'
        if apm1 == "am" or apm1 == "a.m.":
            if len(h) == 1:
                h = "0" + h
        else:
            if int(h) < 4 and apm2 == "at night":
                h = "0" + h
            elif int(h) < 12:
                h = str(int(h) + 12)
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['anchor'] = h + ":" + m
        return output

    def proc_time1(self, direction, sp1, vagueword, sp2, quan, sp3, clockapm, sp4, apm2):
        output = {
            'event_point': '',
            'anchor': '',
            'anchor_mod': '',
            'relation': trans_dir(direction),
            'vagueness': '',
            'rule': 'Time1'
        }
        if vagueword:
            output['vagueness'] = 'yes'
        if quan in quantity_hash_time:
            quan = quantity_hash_time[quan]
        h = ""
        if clockapm == "o'clock":
            if apm2 in ["in the afternoon", "in the evening", "at night"]:
                if re.match(r'0[1-9]', quan):
                    h = quan
                elif int(quan) < 4 and apm2 == "at night":
                    h = "0" + quan
                elif int(quan) < 12:
                    h = str(int(quan) + 12)
                elif quan == "12":
                    h = "00"
                else:
                    h = quan
        else:
            if clockapm == "in the afternoon" or clockapm == "in the evening" or clockapm == "at night":
                if re.match(r'0[1-9]', quan):
                    h = quan
                elif int(quan) < 4 and apm2 == "at night":
                    h = "0" + quan
                elif int(quan) < 12:
                    h = str(int(quan) + 12)
                elif quan == "12":
                    h = "00"
                else:
                    h = quan
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['anchor'] = h + ":00"
        return output

    def proc_time2(self, direction, sp1, vagueword, sp2, qh1, qw1, sp3, m, sp4, past_to, sp5, qw2, sp6, clock, sp7, apm2):
        output = {
            'event_point': '',
            'anchor': '',
            'anchor_mod': '',
            'relation': '',
            'vagueness': '',
            'rule': 'Time2'
        }
        if vagueword:
            output['vagueness'] = 'yes'
        m = transform_num(qh1, qw1)
        if qw2 in quantity_hash_time:
            qw2 = quantity_hash_time[qw2]
        if past_to == "to":
            m = str(60 - int(m))
            if apm2 in ["in the afternoon", "in the evening", "at night"]:
                if int(qw2) < 4 and apm2 == "at night":
                    qw2 = str(int(qw2)-1)
                    qw2 = "0" + qw2
                else:
                    qw2 = str(int(qw2)+11)
            else:
                if (qw2 == 1):
                    qw2 = "00"
                else:
                    qw2 = str(int(qw2)-1)
                    if len(qw2) ==1:
                        qw2 = "0" + qw2
        elif past_to == "past":
            if apm2 in ["in the afternoon", "in the evening", "at night"]:
                if int(qw2) < 4 and apm2 == "at night":
                    qw2 = "0" + qw2
                elif qw2 == "12":
                    qw2 = "00"
                else:
                    qw2 = str(int(qw2)+12)
        if direction == "until":
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['relation'] = trans_dir(direction)
        output['anchor'] = qw2 + ":" + m
        return output

    def proc_time3(self, part1, sp1, direction, sp2, det, sp3, adj, sp4, pod):
        output = {
            'event_point': '',
            'anchor': '',
            'anchor_mod': '',
            'vagueness': '',
            'relation': trans_dir(direction),
            'rule': 'Time3'
        }
        if adj:
            adj = adjective.trans_adj_word1(adjective) + "_"
            if adj == 'current_':
                adj = ''
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['anchor'] = adj + pod
        if part1:
            output['anchor_mod'] = trans_part(part1)
        return output

    def format_date0(self, direction, sp1, monthword, sym1, day, sym2, year):
        output = {
            'event_point': '',
            'event_point2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'relation': '',
            'relation2': '',
            'vagueness': '',
            'rule': 'Date0'
        }
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        if len(monthword) == 1:
            monthword = "0" + monthword
        if len(day) == 1:
            day = "0" + day
        if re.match(r'^[0-1][0-9]$', year):
            year = "20" + year
        elif re.match(r'^[0-9]{2}$', year):
            year = "19" + year
        formatted_date = year + monthword + day
        output['anchor'] = formatted_date
        if direction:
            output['relation'] = trans_dir(direction)
        else:
            output['relation'] = 'equal'
        return output
            
    def format_date1(self, direction, sp1, monthword, sym1, year):
        output = {
            'event_point': '',
            'event_point2': '',
            'anchor': '',
            'anchor2': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'rule': 'Date1'
        }
        if len(monthword) == 1:
            monthword = "0" + monthword
        if re.match(r'^[0-1][0-9]$', year):
            year = "20" + year
        elif re.match(r'^[2-9][0-9]$', year):
            year = "19" + year
        formatted_date = year + monthword
        if re.match(re.compile(preposition.as_of2), direction):
            output['event_point'] = 'start'
            output['event_point2'] = 'finish'
            output['anchor'] = formatted_date + "01"
            output['anchor2'] = month_length(formatted_date)
            output['relation'] = 'equal'
            output['relation2'] = 'before'
        else:
            if direction == 'until':
                output['event_point'] = 'finish'
            else:
                output['event_point'] = 'unspecified'
            output['anchor'] = formatted_date
            output['relation'] = trans_dir(direction)
        return output
        
    def format_date2(self, disc_year, disc_month, direction, sp1, monthword, sp2, sym1, sp3, day):
        output = {
            'event_point': '',
            'event_point2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'relation': '',
            'relation2': '',
            'vagueness': '',
            'rule': 'Date2'
        }
        year = ''
        if is_int(disc_month) and is_int(monthword):
            if int(disc_month) <= 3 and int(monthword) >= 8:
                if is_int(disc_year):
                    year = str(int(disc_year)-1)
            else:
                year = disc_year
        if len(day) == 1:
            day = '0' + day
        if len(monthword) == 1:
            monthword = '0' + monthword
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        formatted = year + monthword + day
        output['anchor'] = formatted
        output['relation'] = trans_dir(direction)
        return output

    def format_date3(self, direction, sp1, monthword, sym1, day, sym2, year):
        output = {
            'event_point': '',
            'event_point2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'rule': 'Date3'
        }
        monthword = month_hash[monthword]
        if day in ord_num_hash:
            day = ord_num_hash[day]
        elif re.match(r'(\d+)(st|nd|rd|th)', day):
            match = re.findall(r'(\d+)(st|nd|rd|th)', day)
            if len(match[0][0]) == 1:
                day = "0" + match[0][0]
            else:
                day = match[0][0]
        formatted_date = year + monthword + day
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['anchor'] = formatted_date
        if direction:
            output['relation'] = trans_dir(direction)
        else:
            output['relation'] = 'equal'
        return output

    def format_date4(self, direction, sp1, monthword, sym1, year):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'event_point': '',
            'event_point2': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'rule': 'Date4'
        }
        month2 = ''
        if monthword in month_hash:
            month2 = month_hash[monthword]
        formatted_date = year + month2
        if re.match(re.compile(preposition.as_of2), direction):
            output['event_point'] = 'start'
            output['event_point2'] = 'finish'
            output['anchor'] = formatted_date + "01"
            output['anchor2'] = month_length(formatted_date)
            output['relation'] = 'equal'
            output['relation2'] = 'before'
        else:
            if direction == 'until':
                output['event_point'] = 'finish'
            else:
                output['event_point'] = 'unspecified'
            output['anchor'] = formatted_date
            output['relation'] = trans_dir(direction)
        return output

    def format_date5(self, disc_year, disc_month, direction, sp1, monthword, sym2, day):
        output = {
            'event_point': '',
            'event_point2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'vagueness': '',
            'relation': trans_dir(direction),
            'relation2': '',
            'rule': 'Date5'
        }
        year = ''
        month2 = ''
        if disc_month in month_hash:
            month2 = month_hash[disc_month]
        else:
            if monthword in month_hash:
                month2 = month_hash[monthword]
        if disc_month and disc_year:
            if is_int(disc_month) and is_int(disc_year):
                if int(disc_month) < 3 and int(month2) > 8:
                    year = str(int(disc_year)-1)
                else:
                    year = disc_year
        day2 = ''
        if day in ord_num_hash:
            day2 = ord_num_hash[day]
        elif re.findall(re.compile('(\d+)[st|nd|rd|th]'), day):
            if len(re.findall(re.compile('(\d+)[st|nd|rd|th]'), day)[0]) == 1:
                day2 = '0' + re.findall(re.compile('(\d+)[st|nd|rd|th]'), day)[0]
            else:
                day2 = re.findall(re.compile('(\d+)[st|nd|rd|th]'), day)[0]
        elif len(day) == 1:
            day2 = '0' + day
        else:
            day2 = day
        if len(month2) == 1:
            month2 = '0' + month2
        formatted_date = year + month2 + day2
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['anchor'] = formatted_date
        return output

    def format_date6(self, disc_year, disc_month, direction, sp1, partword, sp2, monthword):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'relation': '',
            'relation2': '',
            'event_point': '',
            'event_point2': '',
            'vagueness': '',
            'rule': 'Date6'
        }
        part_out = ''
        if partword:
            part_out = trans_part(partword)
        year = ''
        month2 = ''
        if monthword in month_hash:
            month2 = month_hash[monthword]
        if is_int(disc_month) and is_int(monthword):
            if int(disc_month) <= 3 and int(monthword) >= 8:
                if is_int(disc_year):
                    year = str(int(disc_year)-1)
            else:
                year = disc_year
        if len(month2) == 1:
            month2 = '0' + month2
        formatted = year + month2
        if re.match(re.compile(preposition.as_of2), direction):
            if partword:
                if part_out == 'early':
                    output['anchor'] = formatted + '01'
                    output['anchor2'] = formatted
                    output['relation'] = 'equal_or_after'
                    output['relation2'] = 'equal'
                elif part_out == 'late':
                    output['anchor'] = formatted
                    output['anchor2'] = formatted + month_length(formatted)
                    output['relation'] = 'equal'
                    output['relation2'] = 'equal_or_before'
                else:
                    output['anchor'] = formatted
                    output['anchor2'] = formatted
                    output['relation'] = 'equal_or_before'
                    output['relation2'] = 'equal_or_after'
            else:
                output['anchor'] = formatted + '01'
                output['anchor2'] = month_length(formatted)
                output['relation'] = 'equal_or_after'
                output['relation2'] = 'equal_or_before'
        else:
            output['anchor'] = formatted
            output['anchor'] = 'equal'
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        return output

    def format_date7(self, disc_year, disc_month, direction, sp1, det, sp2, day, sp3, of, sp4, monthword):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'relation': '',
            'relation2': '',
            'event_point': '',
            'event_point2': '',
            'vagueness': '',
            'rule': 'Date7'
        }
        year = ''
        month2 = ''
        if monthword in month_hash:
            month2 = month_hash[monthword]
        if is_int(disc_month) and is_int(monthword):
            if int(disc_month) <= 3 and int(monthword) >= 8:
                if is_int(disc_year):
                    year = str(int(disc_year)-1)
            else:
                year = disc_year
        day2 = ''
        if day in ord_num_hash:
            day2 = ord_num_hash[day]
        elif re.match(re.compile('(\d+)(st|nd|rd|th)'), day):
            day2 = re.findall(re.compile('(\d+)(st|nd|rd|th'), day)[0][0]
        else:
            day2 = day
        if len(day2) == 1:
            day2 = '0' + day2
        if len(month2) == 1:
            month2 = '0' + month2
        formatted = year + month2 + day2
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        output['anchor'] = formatted
        if direction:
            output['relation'] = trans_dir(direction)
        else:
            output['relation'] = 'equal'
        return output

    def format_date8(self, disc_year, disc_month, direction, sp1, partword, sp2, adj, sp3, unit):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'event_point': '',
            'event_point2': '',
            'relation': '',
            'relation2': '',
            'vagueness': '',
            'rule': 'Date8'
        }
        month2 = ''
        year2 = ''
        part_out = ''
        if partword:
            part_out = trans_part(partword)
        if unit == 'month':
            if re.match(re.compile(adjective.before_adj), adj):
                if disc_month == 1:
                    month2 = '12'
                    if is_int(disc_year):
                        year2 = str(int(disc_year)-1)
                else:
                    if is_int(disc_month):
                        month2 = str(int(disc_month-1))
                    year2 = disc_year
            elif re.match(re.compile(adjective.after_adj), adj):
                if disc_month == '12':
                    month2 = '1'
                    if is_int(disc_year):
                        year2 = str(int(disc_year)+1)
                else:
                    if is_int(disc_month):
                        month2 = str(int(disc_month) + 1)
                    year2 = disc_year
            else:
                year2 = disc_year
                month2 = disc_month
        else:
            if adj:
                if unit in month_hash:
                    month2 = month_hash[unit]
                elif re.match(re.compile(adjective.before_adj), adj):
                    if is_int(disc_year):
                        year2 = str(int(disc_year)-1)
                elif re.match(re.compile(adjective.after_adj), adj):
                    if is_int(disc_year):
                        year2 = str(int(disc_year)+1)
                else:
                    year2 = disc_year
        if len(month2) == 1:
            month2 = '0' + month2
        formatted = year2 + month2
        if unit == 'year':
            if re.match(re.compile(preposition.as_of2), direction):
                if part:
                    if part_out == 'early':
                        output['anchor'] = year2 + '0101'
                        output['anchor2'] = year2
                    elif part_out == 'late':
                        output['anchor'] = year2 + '1231'
                        output['anchor2'] = year2
                    else:
                        output['anchor'] = year2
                        output['anchor2'] = year2
                else:
                    output['anchor'] = year2 + '0101'
                    output['anchor2'] = year2 + '1231'
            else:
                if part:
                    if part_out == 'early':
                        output['anchor'] = year2 + '0101'
                        output['anchor2'] = year2
                        output['relation'] = 'equal_or_before'
                        output['relation2'] = 'equal'
                    elif part_out == 'late':
                        output['anchor'] = year2 + '1231'
                        output['anchor2'] = year2
                        output['relation'] = 'equal_or_before'
                        output['relation2'] = 'equal'
                else:
                    output['anchor'] = year2
        else:
            if re.match(re.compile(preposition.as_of2), direction):
                if part:
                    if part_out == 'early':
                        output['anchor'] = formatted + '01'
                        output['anchor'] = formatted
                    elif part_out == 'late':
                        output['anchor'] = formatted
                        output['anchor2'] = month_length(formatted)
                    else:
                        output['anchor'] = formatted
                        output['anchor2'] = formatted
                else:
                    output['anchor'] = formatted + '01'
                    output['anchor2'] = month_length(formatted)
            else:
                output['anchor'] = formatted
        if direction == 'until':
            output['event_point'] = 'finish'
        else:
            output['event_point'] = 'unspecified'
        return output

    def format_date9(self, direction, sp1, vagueword, sp2, part1, sp3, year):
        direction_out = ''
        part1_out = ''
        anchor = ''
        anchor2 = ''
        if part1 > 0:
            part1_out = trans_part(part1)
        formatted_date = year
        if re.match(re.compile(preposition.as_of2), direction):
            if part1:
                if part1_out == 'early':
                    anchor = year + "0101"
                    anchor2 = year
                elif part1_out == 'late':
                    anchor = year
                    anchor2 = year + "1231"
                else:
                    anchor = year
                    anchor2 = year
            else:
                anchor = year + '0101'
                anchor2 = year + '1231'
        else:
            anchor = year
        output = self.time_target1(direction, part1, anchor, anchor2)
        if vagueword:
            output['vagueness'] = 'yes'
        output['rule'] = 'Date9'
        return output

    def format_date10(self, disc_year, disc_month,  direction, sp1, vagueword, sp2, det, sp3, adj, sp4, unit):
        anchor = ''
        anchor2 = ''
        month2 = ''
        year2 = ''
        if unit == 'month':
            if re.match(re.compile(adjective.before_adj), adj):
                if disc_month == '1':
                    month2 = '12'
                    if is_int(disc_year):
                        year2 = str(int(disc_year)-1)
                else:
                    if is_int(disc_month):
                        month2 = str(int(disc_month)-1)
                    year2 = disc_year
            elif re.match(re.compile(adjective.after_adj), adj):
                if disc_month == '12':
                    month2 = '1'
                    if is_int(disc_year):
                        year2 = str(int(disc_year)+1)
                else:
                    if is_int(disc_month):
                        month2 = str(int(disc_month)+1)
                    year2 = disc_year
            else:
                year2 = disc_year
                month2 = disc_month
        else:
            if unit in month_hash:
                month2 = month_hash[unit]
            if adj:
                if re.match(re.compile(adjective.before_adj), adj):
                    if is_int(disc_year):
                        year2 = str(int(disc_year)-1)
                elif re.match(re.compile(adjective.after_adj), adj):
                    if is_int(disc_year):
                        year2 = str(int(disc_year)+1)
                else:
                    year2 = disc_year
        if len(month2) == 1:
            month2 = '0' + month2
        formatted = year2 + month2
        if unit == 'year':
            if re.match(re.compile(preposition.as_of2), direction):
                anchor = year2 + '0101'
                anchor2 = year2 + '1231'
            else:
                anchor = year2
        else:
            if re.match(re.compile(preposition.as_of2), direction):
                anchor = formatted + '01'
                anchor2 = month_length(formatted)
            else:
                anchor = formatted
        partword = ''
        output = self.time_target1(direction, partword, anchor, anchor2)
        if vagueword:
            output['vagueness'] = 'yes'
        output['rule'] = 'Date10'
        return output

    def format_date11(self, direction, sp1, det1, sp2, part1, sp3, decade):
        output = {
            'anchor': '',
            'anchor2': '',
            'relation': 'equal_or_after',
            'relation2': 'equal_or_before',
            'event_point': 'start',
            'event_point2': 'finish',
            'anchor_mod': '',
            'anchor_mod2': '',
            'vagueness': '',
            'rule': 'Date11'
        }
        value = ''
        part_out = ''
        if re.match(r'(19)(\d)(0\'?s)', decade):
            match = [x for x in re.findall(r'(19)(\d)(0\'?s)', decade)]
            value = "19" + match[0][1]
        else:
            value = "19" + decade_hash[decade[0:4]]
        if len(part1) > 0:
            part_out = trans_part(part1)
        if re.match(re.compile(preposition.as_of2), direction):
            if len(part1) > 0:
                if part_out == 'early':
                    output['anchor'] = value + '00101'
                    output['anchor2'] = value
                elif part_out == 'late':
                    output['anchor'] = value
                    output['anchor2'] = value + '01231'
                else:
                    output['anchor'] = value
                    output['anchor2'] = value
            else:
                output['anchor'] = value + '00101'
                output['anchor2'] = value + '01231'
        else:
            output['anchor'] = value
        return output

    def time_target1(self, direction, part1, anchor, anchor2):
        output = {
            'event_point': '',
            'event_point2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'relation': '',
            'relation2': '',
            'anchor': anchor,
            'anchor2': anchor2,
            'vagueness': ''
        }
        if len(part1) > 0:
            part1_out = trans_part(part1)
        if re.match(re.compile(preposition.as_of2), direction):
            output['event_point'] = 'start'
            output['event_point2'] = 'finish'
            if len(part1) > 0:
                if part1_out == 'early':
                    output['relation'] = 'equal_or_after'
                    output['anchor_mod2'] = 'early'
                    output['relation2'] = 'equal'
                elif part1_out == 'late':
                    output['anchor_mod'] = 'late'
                    output['relation'] = 'equal'
                    output['relation2'] = 'equal_or_after'
                else:
                    output['anchor_mod'] = 'early'
                    output['relation'] = 'equal'
                    output['anchor_mod2'] = 'late'
                    output['relation2'] = 'equal'
            else:
                output['relation'] = 'equal'
                output['relation2'] = 'equal_or_after'
        else:
            if direction == "until":
                output['event_point'] = 'finish'
            else:
                output['event_point'] = 'unspecified'
            if len(part1) > 0:
                output['anchor_mod'] = trans_part(part1)
            if len(direction) > 0:
                output['relation'] = trans_dir(direction)
            else:
                output['relation'] = 'equal'
        return output

    def format_date12(self, direction, sp1, monthword, sp2, num1, sp3, to, sp4, the, sp5, num2):
        output = {
            'anchor': '',
            'anchor2': '',
            'relation': '',
            'relation2': '',
            'event_point': '',
            'event_point2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'vagueness': '',
            'rule': 'Date12'
        }
        #if to == 'and':
        #    output['text'] = " ".join([direction, monthword, num1, "and", monthword, num2])
        #elif direction == 'between':
        #    output['text'] = " ".join(["between", monthword, num1, "and", monthword, num2])
        #else:
        #    output['text'] = " ".join(["from", monthword, num1, "to", monthword, num2])
        return output
    
    def format_duration0(self, durword, sp1, vagueword, sp2, det, sp3, adj, sp4, num1, sp5, dash, sp6, num2, sp7, tunit, s):
        output = self.dur_target0(durword, vagueword, adj, num1, tunit, num2, tunit, 'Duration0')
        return output

    def format_duration1(self, durword, sp1, vagueword, sp2, det, sp3, adj, sp4, qh1, qw1, qt1, sp5, tunit1, s1, sp6, to, sp7, qh2, qw2, qt2, sp8, tunit2, s2):
        quant1 = transform_num2(qh1, qw1, qt1)
        quant2 = transform_num2(qh2, qw2, qt2)
        output = self.dur_target0(durword, vagueword, adj, quant1, tunit1, quant2, tunit2, 'Duration1')
        return output

    def format_duration2(self, durword, sp1, det, sp2, adj, sp3, adjnum, sp4, tunit, s):
        num2 = ''
        tunit2 = ''
        vagueword = ''
        output = self.dur_target0(durword, vagueword, adj, adjnum, tunit, num2, tunit2, 'Duration2')
        return output

    def format_duration3(self, durword, sp1, vagueword, sp2, pdir, sp3, det, sp4, adj, sp5, qh, qw, qt, sp6, tunit, s):
        quantout = transform_num2(qh, qw, qt)
        output = self.dur_target1(durword, vagueword, pdir, adj, quantout, tunit, "Duration3")
        return output

    def format_duration4(durword, sp1, det, sp2, adj, sp3, tunit, s):
        quantout = 'plural'
        vagueword = ''
        pdir = ''
        output = self.dur_target2(durword, vagueword, pdir, adj, quantout, tunit, "Duration4")
        return output

    def format_duration5(self, with1, sp1, a, num1, sp2, dash, sp3, num2, sp4, tunit, s, sp5, long1, history, of):
        output = {
            'plan': '',
            'relation': 'equal',
            'relation2': '',
            'event_point': 'start',
            'event_point2': '',
            'vagueness': '',
            'direction': 'minus',
            'quan_mod':  '',
            'interval_op': 'jump',
            'anchor': 'event',
            'anchor2': '',
            'anchor_point': 'finish',
            'rule': 'Duration5'
        }
        if re.match('to\s{1,3}\w+', with1):
            output['plan'] = 'yes'
        return output

    def format_duration6(self, with1, sp1, a, qh1, qw1, qt1, sp2, tunit1, s1, sp3, to, sp4, qh2, qw2, qt2, sp5, tunit2, s2, sp6, long1, history, of):
        output = {
            'relation': 'equal',
            'relation2': '',
            'direction': 'minus',
            'interval_op': 'jump',
            'vagueness': '',
            'event_point': '',
            'event_point2': '',
            'quan_mod':  '',
            'anchor': 'event',
            'anchor2': '',
            'anchor_point': 'finish',
            'plan': '',
            'rule': 'Duration6'
        }
        return output

    def format_duration7(self, with1, sp1, a, adjnum, sp2, dash, sp3, tunit, s, sp4, long1, history, of):
        output = {
            'relation': 'equal',
            'relation2': '',
            'direction': 'minus',
            'interval_op': 'jump',
            'vagueness': '',
            'event_point': '',
            'event_point2': '',
            'quan_mod':  '',
            'anchor': 'event',
            'anchor2': '',
            'anchor_point': 'finish',
            'plan': '',
            'rule': 'Duration7'
        }
        return output

    def format_duration8(self, with1, sp1, a, vagueword, sp2, pdir, sp3, qh, qw, qt, sp4, dash, sp5, tunit, s, sp6, long1, history, of):
        output = {
            'relation': '',
            'relation2': '',
            'direction': 'minus',
            'event_point': '',
            'event_point2': '',
            'vagueness': '',
            'interval_op': 'jump',
            'anchor': 'event',
            'anchor2': '',
            'quan_mod':  '',
            'anchor_point': '',
            'plan': '',
            'rule': 'Duration8'
        }
        if vagueword:
            output['vagueness'] = 'yes'
        if pdir:
            if pdir in pdir_hash:
                output['relation'] = pdir_hash[pdir]
            else:
                output['relation'] = 'equal'
        else:
            output['relation'] = 'equal'
        return output

    def format_duration9(self, vagueword, sp1, det, sp2, adj, sp3, num1, sp4, dash, sp5, num2, sp6, tunit, s, sp7, prep):
        output = {
            'relation': 'equal_or_after',
            'relation2': 'equal_or_before',
            'interval_op': 'jump',
            'vagueness': '',
            'direction': '',
            'anchor': '',
            'anchor2': '',
            'anchor_point': '',
            'event_point': '',
            'event_point2': '',
            'quan_mod':  '',
            'plan': '',
            'rule': 'Duration9'
        }
        if prep in preposition.after_prep:
            output['event_point'] = 'after'
            output['direction'] = 'plus'
        else:
            output['event_point'] = 'before'
            output['direction'] = 'minus'
        if vagueword:
            output['vagueness'] = 'yes'
        if re.match(re.compile('[\.\?\!\:\;\,]'), sp7):
            output['anchor'] = 'unspecified'
        else:
            output['anchor'] = 'event'
        return output

    def format_duration10(self, vagueword, sp1, det, sp2, adj, sp3, qh1, qw1, qt1, sp4, tunit1, s1, sp5, to, sp6, qh2, qw2, qt2, sp7, tunit2, s2, sp8, prep, sp9):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_point': '',
            'direction': '',
            'relation': 'equal_or_after',
            'relation2': 'equal_or_before',
            'interval_op': 'jump',
            'vagueness': '',
            'event_point': '',
            'event_point2': '',
            'quan_mod':  '',
            'plan': '',
            'rule': 'Duration10'
        }
        tunitout1 = ''
        if tunit1:
            tunitout1 = tunit1
        else:
            tunitout1 = tunit2
        tunitout2 = tunit2
        if re.match(re.compile(preposition.after_prep), prep):
            output['event_point'] = 'after'
            output['direction'] = 'plus'
        else:
            output['event_point'] = 'before'
            output['direction'] = 'minus'
        if vagueness:
            output['vagueness'] = 'yes'
        if sp9 in ['.', '?', '!', ':', ';', ',']:
            output['anchor'] = 'unspecified'
        else:
            output['anchor'] = 'event'
        return output

    def format_duration11(self, det, sp1, adj, sp2, adjnum, sp3, tunit, s, sp4, prep, sp5):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_point': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'interval_op': 'drag',
            'direction': '',
            'plan': '',
            'quan_mod':  '',
            'event_point': '',
            'event_point2': '',
            'rule': 'Duration11'
        }
        if re.match(re.compile(preposition.after_prep), prep):
            output['event_point'] = 'after'
            output['direction'] = 'plus'
        else:
            output['event_point'] = 'before'
            output['direction'] = 'minus'
        if sp5 in ['.', '?', '!', ':', ';', ',']:
            output['anchor'] = 'unspecified'
        else:
            output['anchor'] = 'event'
        return output

    def format_duration12(self, vagueword, sp1, pdir, sp2, det, sp3, adj, sp4, qh, qw, qt, sp5, tunit, s, sp6, prep, sp7):
        output = {
            'anchor': '',
            'anchor2': '',
            'anchor_point': '',
            'event_point': '',
            'event_point2': '',
            'relation': '',
            'relation2': '',
            'direction': '',
            'interval_op': 'drag',
            'vagueness': '',
            'quan_mod':  '',
            'plan': '',
            'rule': 'Duration12'
        }
        if re.match(re.compile(preposition.after_prep), prep):
            output['event_point'] = 'after'
            output['direction'] = 'plus'
        else:
            output['event_point'] = 'before'
            output['direction'] = 'minus'
        if vagueword:
            output['vagueness'] = 'yes'
        if pdir in pdir_hash:
            pdir2 = pdir_hash[pdir]
        else:
            pdir2 = ''
        if pdir:
            output['quan_mod'] = pdir2
        if sp7 in ['.', '?', '!', ':', ';', ',']:
            output['anchor'] = 'unspecified'
        else:
            output['anchor'] = 'event'
        return output

    def format_duration13(self, partword, sp1, direction, sp2, det, sp3, adj, sp4, tunit, s):
        output = {
            'anchor': '',
            'anchor2': '',
            'relation': '',
            'relation2': '',
            'event_point': '',
            'event_point2': '',
            'anchor_mod': '',
            'anchor_mod2': '',
            'quan_mod':  '',
            'rule': 'Duration13'
        }
        adj2 = trans_adj_word1(adj)
        if adj2 == 'current':
            adj2 = ''
        if adj2 > 0:
            output['anchor'] = adj2 + "_" + tunit
        else:
            output['anchor'] = tunit
        if re.match(re.compile(preposition.as_of3), direction):
            output['event_point'] = 'start'
            output['anchor_mod'] = 'start'
            output['relation'] = 'equal'
            output['event_point2'] = 'finish'
            if adj2:
                output['anchor2'] = adj2 + "_" + tunit
            else:
                output['anchor2'] = tunit
            output['anchor_mod2'] = 'finish'
            if re.match(re.compile(adjective.before_adj), adj):
                output['relation2'] = 'equal_or_after'
            else:
                output['relation2'] = 'equal'
        else:
            if direction == 'until':
                output['event_point'] = 'finish'
            else:
                output['event_point'] = 'unspecified'
            if direction:
                output['relation'] = trans_dir(direction)
            else:
                output['relation'] = 'equal'
            if partword:
                output['anchor_mod'] = trans_part2(partword)
        return output

    def dur_target0(self, durword, vagueword, adj, num1, tunit1, num2, tunit2, rule):
        output = {
            'event_point': '',
            'event_point2': '',
            'direction': '',
            'interval_op': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_point': '',
            'quan_mod':  '',
            'rule': rule,
            'plan': ''
        }
        quant1out = num1
        quant1Bout = ''
        tunitout1 = tunit1
        tunitout1B = ''
        quant2out = num1
        quant2Bout = ''
        tunitout2 = tunit2
        tunitout2B = ''
        if len(num2) > 0:
            quant1bout = num2
            tunitout1b = tunit1
            quant2Bout = num2
            tunit2b = tunit2
        if len(vagueword) > 0:
            output['vagueness'] = 'yes'
        if re.match(re.compile('in|within|during'), durword):
            if len(adj) > 0:
                output['direction'] = trans_adj_word2(adj)
            else:
                output['direction'] = 'plus'
            output['event_point'] = 'unspecified'
            output['event_point2'] = 'unspecified'
            output['interval_op'] = 'drag'
            output['relation'] = 'equal'
            output['anchor'] = 'narrative_reference'
        elif adj:
            output['event_point'] = 'start'
            output['event_point2'] = 'finish'
            output['anchor'] = 'narrative_reference'
            output['anchor2'] = 'narrative_reference'
            if re.match(re.compile(adjective.before_adj), adj):
                output['direction'] = trans_adj_word2(adj)
                output['interval_op'] = 'jump'
                output['relation'] = 'equal'
                output['relation2'] = 'equal_or_after'
            else:
                output['relation'] = 'equal'
                output['direction'] = trans_adj_word2(adj)
                output['interval_op'] = 'jump'
                output['relation2'] = 'equal'
        else:
            output['relation'] = 'equal'
            output['event_point'] = 'start'
            output['direction'] = 'minus'
            output['inverval_op'] = 'jump'
            output['anchor'] = 'event'
            output['anchor_point'] = 'finish'
        return output

    def dur_target1(self, durword, vagueword, pdir, adj, num, tunit, rule):
        output = {
            'event_point': '',
            'event_point2': '',
            'direction': '',
            'interval_op': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_point': '',
            'rule': rule,
            'quan_mod':  '',
            'plan': ''
        }
        quant1out = num
        tunitout1 = tunit
        quant2out = num
        tunitout2 = tunit
        if len(vagueword) > 0:
            output['vagueness'] = 'yes'
        if re.match(re.compile(preposition.within), durword):
            if len(adj) > 0:
                output['direction'] = trans_adj_word2(adj)
            else:
                output['direction'] = 'plus'
            output['event_point'] = 'unspecified'
            output['interval_op'] = 'drag'
            if len(pdir) > 0:
                if re.match(re.compile(adjective.before_adj), adj):
                    output['relation'] = self.before_pdir_hash[pdir]
                elif re.match(adjective.after_adj_re, adj):
                    output['relation'] = self.after_pdir_hash[pdir]
                else:
                    output['relation'] = 'equal'
            else:
                output['relation'] = 'equal'
            output['anchor'] = 'narrative_reference'
        elif adj:
            output['event_point'] = 'start'
            output['event_point2'] = 'finish'
            output['anchor'] = 'narrative_reference'
            output['anchor2'] = 'narrative_reference'
            if re.match(adjective.before_adj, adj):
                output['direction'] = trans_adj_word2(adj)
                output['interval_op'] = 'jump'
                if len(pdir) > 0:
                    output['relation'] = self.before_pdir_hash[pdir]
                else:
                    output['relation'] = 'equal'
                output['relation2'] = 'equal_or_after'
            else:
                output['relation'] = 'equal'
                output['relation2'] = trans_adj_word2(adj)
                output['interval_op'] = 'jump'
                if len(pdir) > 0:
                    output['relation2'] = self.after_pdir_hash[pdir]
                else:
                    output['relation2'] = 'equal'
        else:
            if len(pdir) > 0:
                output['relation'] = self.dur_pdir_hash[pdir]
            else:
                output['relation'] = 'equal'
            # if output['relation'] == # i don't really know what's going on here
            output['event_point'] = 'unspecified'
            output['event_point2'] = 'unspecified'
            output['direction'] = 'minus'
            output['interval_op'] = 'jump'
            output['anchor'] = 'event'
            output['anchor_point'] = 'finish'
        return output

    def dur_target2(self, durword, vagueword, pdir, adj, num, tunit):
        output = {
            'event_point': '',
            'event_point2': '',
            'direction': '',
            'direction2': '',
            'interval_op': '',
            'interval_op2': '',
            'vagueness': '',
            'relation': '',
            'relation2': '',
            'anchor': '',
            'anchor2': '',
            'anchor_point': ''
        }
        quant1out = num
        tunitout1 = tunit
        quant2out = num
        tunitout2 = tunit
        if len(vagueword) > 0:
            output['vagueness'] = 'yes'
        if re.match(within_re, durword):
            if len(adj) > 0:
                output['direction'] = trans_adj_word2(adj)
            else:
                output['direction'] = 'plus'
            output['event_point'] = 'unspecified'
            output['interval_op'] = 'drag'
            if len(pdir) > 0:
                if re.match(re.compile(adjective.before_adj), adj):
                    output['relation'] = before_pdir_hash[pdir]
                elif re.match(adjective.after_adj_re, adj):
                    output['relation'] = after_pdir_hash[pdir]
                else:
                    output['relation'] = 'equal'
            else:
                output['relation'] = 'equal'
            output['anchor'] = 'narrative_reference'
        elif not adj == "":
            output['event_point'] = 'start'
            output['event_point2'] = 'finish'
            output['anchor'] = 'narrative_reference'
            output['anchor2'] = 'narrative_reference'
            if re.match(adjective.before_adj, adj):
                output['direction'] = trans_adj_word2(adj)
                output['interval_op'] = 'jump'
                if len(pdir) > 0:
                    output['relation'] = before_pdir_hash[pdir]
                else:
                    output['relation'] = 'equal'
                output['relation2'] = 'equal_or_after'
            else:
                output['relation'] = 'equal'
                output['relation2'] = trans_adj_word2(adj)
                output['interval_op2'] = 'jump'
                if len(pdir) > 0:
                    output['relation2'] = after_pdir_hash[pdir]
                else:
                    output['relation2'] = 'equal'
        else:
            if len(pdir) > 0:
                output['relation'] = dur_pdir_hash[pdir]
            else:
                output['relation'] = 'equal'
            # if output['relation'] == # i don't really know what's going on here
            output['direction'] = 'minus'
            output['interval_op'] = 'jump'
            output['anchor'] = 'event'
            output['anchor_point'] = 'finish'
        return output

    def find_temporal(self, sentence):
        times, dates, durations = [], [], []
        time = self.find_time(sentence)
        date = self.find_date(sentence)
        duration = self.find_duration(sentence)
        if time:
            times.append(time)
        if date:
            dates.append(date)
        if duration:
            durations.append(duration)
        sentence['times'] = times
        sentence['dates'] = dates
        sentence['durations'] = durations

    def find_time(self, sentence):
        text = sentence['text']
        word_list = sentence['tokens']
        len_word_list = len(word_list)
        time = {}
        for r, rule in enumerate(self.time_rule_list):
            m = rule.findall(text)
            for match in m:
                items = [t for t in match]
                str_m = match[0].lower().strip()
                len_str_m = len(tokenize(str_m))
                for i in range(len_word_list):
                    if i < len_word_list - len_str_m + 1:
                        str_words = ' '.join(word_list[i:i + len_word_list]).lower().strip()
                        if str_m == str_words and str_m not in (' ', ''):
                            attrib = sentence['attrib']
                            attrib['WordCount'] = str(i + 1)
                            attrib['STR'] = str_m
                            time = {'attrib': attrib}
                            if r == 0:
                                direction, sp1, vagueword, sp2, h, colon, m, sp3, apm1, sp4, apm2 = items
                                t = self.proc_time0(direction, sp1, vagueword, sp2, h, colon, m, sp3, apm1, sp4, apm2)
                                time['attrib'].update(t)
                                time['text'] = ''.join(items)
                            elif r == 1:
                                direction, sp1, vagueword, sp2, quan, sp3, clockapm, sp4, apm2 = items
                                t = self.proc_time1(direction, sp1, vagueword, sp2, quan, sp3, clockapm, sp4, apm2)
                                time['attrib'].update(t)
                                time['text'] = ''.join(items)
                            elif r == 2:
                                direction, sp1, vagueword, sp2, qh1, qw1, sp3, m, sp4, past_to, sp5, sq2, sp6, clock, sp7, apm1 = items
                                t = self.proc_time2(direction, sp1, vagueword, sp2, qh1, qw1, sp3, m, sp4, past_to, sp5, sq2, sp6, clock, sp7, apm1)
                                time['attrib'].update(t)
                                time['text'] = ''.join(items)
                            elif r == 3:
                                part1, sp1, direction, sp2, det, sp3, adj4, sp4, pod = items
                                t = self.proc_time3(part1, sp1, direction, sp2, det, sp3, adj, sp4, pod)
                                time['attrib'].update(t)
                                time['text'] = ''.join(items)
        return time

    def find_date(self, sentence):
        text = sentence['text']
        word_list = sentence['tokens']
        len_word_list = len(word_list)
        date = {}
        for r, rule in enumerate(self.date_rule_list):
            m = rule.findall(text)
            for match in m:
                items = [t for t in match]
                str_m = ' '.join(tokenize(' '.join(items)))
                len_str_m = len(tokenize(str_m))
                for i in range(len_word_list):
                    if i < len_word_list - len_str_m + 1:
                        str_words = ' '.join(word_list[i:i + len_str_m]).lower().strip()
                        if str_m == str_words and str_m not in (' ', ''):
                            attrib = dict(sentence['attrib'])
                            attrib['WordCount'] = str(i + 1)
                            attrib['STR'] = str_m
                            date = {'attrib': attrib}
                            if r == 0:
                                direction, sp1, monthword, sym1, day, sym2, year = items
                                d = self.format_date0(direction, sp1, monthword, sym1, day, sym2, year)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 1:
                                direction, sp1, monthword, sym1, year = items
                                d = self.format_date1(direction, sp1, monthword, sym1, year)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 2:
                                direction, sp1, monthword, sp2, sym1, sp3, day = items
                                d = self.format_date2('2015', '06', direction, sp1, monthword, sp2, sym1, sp3, day)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 3:
                                direction, sp1, monthword, sym1, day, sym2, year = items
                                d = self.format_date3(direction, sp1, monthword, sym1, day, sym2, year)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 4:
                                direction, sp1, monthword, sym1, year = items
                                d = self.format_date4(direction, sp1, monthword, sym1, year)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 5:
                                direction, sp, monthword, sym1, day = items
                                d = self.format_date5(direction, sp, monthword, sym1, day)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 6:
                                direction, sp1, partword, sp2, monthword = items
                                d = self.format_date6(direction, sp1, partword, sp2, monthword)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 7:
                                direction, sp1, det, sp2, day, sp3, of, sp4, monthword = items
                            elif r == 8:
                                direction, sp1, partword, sp2, adj, sp3, unit = items
                            elif r == 9:
                                direction, sp1, vagueword, sp2, part1, sp3, year = items
                                d = self.format_date9(direction, sp1, vagueword, sp2, part1, sp3, year)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 10:
                                direction, sp1, vagueword, sp2, det, sp3, adj, sp4, unit = items
                            elif r == 11:
                                direction, sp1, det1, sp2, part1, sp3, decade = items
                                d = self.format_date11(direction, sp1, det1, sp2, part1, sp3, decade)
                                date['attrib'].update(d)
                                date['text'] = ''.join(items)
                            elif r == 12:
                                direction, sp1, monthword, sp2, num1, sp3, to, sp4, the, sp5, num2 = items
        return date

    def find_duration(self, sentence):
        text = sentence['text']
        word_list = sentence['tokens']
        len_word_list = len(word_list)
        duration = {}
        for r, rule in enumerate(self.duration_rule_list):
            m = rule.findall(text)
            for match in m:
                items = [t for t in match]
                str_m = ' '.join(tokenize(' '.join(items)))
                len_str_m = len(tokenize(str_m))
                for i in range(len_word_list):
                    if i < len_word_list - len_str_m + 1:
                        str_words = ' '.join(word_list[i:i + len_str_m]).lower().strip()
                        if str_m == str_words and str_m not in (' ', ''):
                            attrib = dict(sentence['attrib'])
                            attrib['WordCount'] = str(i + 1)
                            attrib['STR'] = str_m
                            duration = {'attrib': attrib}
                            if r == 0:
                                durword, sp1, vagueword, sp2, det, sp3, adj, sp4, num1, sp5, dash, sp6, num2, sp7, tunit, s = items
                                d = self.format_duration0(durword, sp1, vagueword, sp2, det, sp3, adj, sp4, num1, sp5, dash, sp6, num2, sp7, tunit, s)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 1:
                                durword, sp1, vagueword, sp2, det, sp3, adj, sp4, qh1, qw1, qt1, sp5, tunit1, s1, sp6, to, sp7, qh2, qw2, qt2, sp8, tunit2, s2 = items
                                d = self.format_duration1(durword, sp1, vagueword, sp2, det, sp3, adj, sp4, qh1, qw1, qt1, sp5, tunit1, s1, sp6, to, sp7, qh2, qw2, qt2, sp8, tunit2, s2)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 2:
                                durword, sp1, det, sp2, adj, sp3, adjnum, sp4, tunit, s = items
                                d = self.format_duration2(durword, sp1, det, sp2, adj, sp3, adjnum, sp4, tunit, s)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 3:
                                durword, sp1, vagueword, sp2, pdir, sp3, det, sp4, adj, sp5, qh, qw, qt, sp6, tunit, s = items
                                d = self.format_duration3(durword, sp1, vagueword, sp2, pdir, sp3, det, sp4, adj, sp5, qh, qw, qt, sp6, tunit, s)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 4:
                                durword, sp1, det, sp2, adj, sp3, tunit, s = items
                                d = self.format_duration4(durword, sp1, det, sp2, adj, sp3, tunit, s)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 5:
                                with1, sp1, a, num1, sp2, dash, sp3, num2, sp4, tunit, s, sp5, long1, history, of = items
                                d = self.format_duration5(with1, sp1, a, num1, sp2, dash, sp3, num2, sp4, tunit, s, sp5, long1, history, of)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 6:
                                with1, sp1, a, qh1, qw1, qt1, sp2, tunit1, s1, sp3, to, sp4, qh2, qw2, qt2, sp5, tunit2, s2, sp6, long1, history, of = items
                                d = self.format_duration6(with1, sp1, a, qh1, qw1, qt1, sp2, tunit1, s1, sp3, to, sp4, qh2, qw2, qt2, sp5, tunit2, s2, sp6, long1, history, of)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 7:
                                with1, sp1, a, adjnum, sp2, dash, sp3, tunit, s, sp4, long1, history, of = items
                                d = self.format_duration7(with1, sp1, a, adjnum, sp2, dash, sp3, tunit, s, sp4, long1, history, of)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 8:
                                with1, sp1, a, vagueword, sp2, pdir, sp3, qh, qw, qt, sp4, dash, sp5, tunit, s, sp6, long1, history, of = items
                                d = self.format_duration8(with1, sp1, a, vagueword, sp2, pdir, sp3, qh, qw, qt, sp4, dash, sp5, tunit, s, sp6, long1, history, of)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 9:
                                vagueword, sp1, det, sp2, adj, sp3, num1, sp4, dash, sp5, num2, sp6, tunit, s, sp7, prep = items
                                d = self.format_duration9(vagueword, sp1, det, sp2, adj, sp3, num1, sp4, dash, sp5, num2, sp6, tunit, s, sp7, prep)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 10:
                                vagueword, sp1, det, sp2, adj, sp3, qh1, qw1, qt1, sp4, tunit1, s1, sp5, to, sp6, qh2, qw2, qt2, sp7, tunit2, s2, sp8, prep, sp9 = items
                                d = self.format_duration10(vagueword, sp1, det, sp2, adj, sp3, qh1, qw1, qt1, sp4, tunit1, s1, sp5, to, sp6, qh2, qw2, qt2, sp7, tunit2, s2, sp8, prep, sp9)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 11:
                                det, sp1, adj, sp2, adjnum, sp3, tunit, s, sp4, prep, sp5 = items
                                d = self.format_duration11(det, sp1, adj, sp2, adjnum, sp3, tunit, s, sp4, prep, sp5)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 12:
                                vagueword, sp1, pdir, sp2, det, sp3, adj, sp4, qh, qw, qt, sp5, tunit, s, sp6, prep, sp7 = items
                                d = self.format_duration12(vagueword, sp1, pdir, sp2, det, sp3, adj, sp4, qh, qw, qt, sp5, tunit, s, sp6, prep, sp7)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
                            elif r == 13:
                                partword, sp1, direction, sp2, det, sp3, adj, sp4, tunit, s = items
                                d = self.format_duration13(partword, sp1, direction, sp2, det, sp3, adj, sp4, tunit, s)
                                duration['attrib'].update(d)
                                duration['text'] = ''.join(items)
        return duration

    @staticmethod
    def within_scope(target_pos, temporal_pos, temporal_len, tokens):
        window = 8 # ?
        if target_pos > temporal_pos:
            scope = target_pos - temporal_pos
        else:
            scope = temporal_pos - target_pos
        if (scope <= window) and (scope > (0 - temporal_len)):
            return True
        else:
            return False

    #@staticmethod
    #def trans_ref_event(event):
    #    if re.match(re.compile(ref_event.admission), event):
    #        return 'admission'
    #    elif re.match(re.compile(ref_event.discharge), event):
    #        return 'discharge'
    #    elif re.match(re.compile(ref_event.transfer), event):
    #        return 'transfer'
    #    elif re.match(re.compile(ref_event.operation), event):
    #        return 'operation'
    #    elif re.match(re.compile(ref_event.postoperation), event):
    #        return 'postoperation'
    #    elif re.match(re.compile(ref_event.hospital_day), event):
    #        return 'hospitalization'
    #    elif re.match(re.compile(ref_event.postpartum), event):
    #        return 'postpartum'
    #    else:
    #        return event

    #@staticmethod
    #def trans_dir(string):
    #    if re.match(re.compile(after_prep), string):
    #        return "after"
    #    elif re.match(re.compile(eq_after), string):
    #        return "equal_or_after"
    #    elif re.match(re.compile(before_prep), string):
    #        return "before"
    #    elif re.match(re.compile(eq_before), string):
    #        return "equal_or_before"
    #    elif re.match(re.compile(within), string) or re.match(re.compile(simul_prep1), string) or re.match(re.compile(simul_prep2), string) or re.match(re.compile(simul_prep3), string) or re.match(re.compile(as_of1), string) or re.match(re.compile(r'until'), string):
    #        return "equal"

    #@staticmethod
    #def trans_part(string):
    #    if re.match(re.compile(start_part), string):
    #        return "early"
    #    elif re.match(re.compile(mid_part), string):
    #        return "mid"
    #    elif re.match(re.compile(end_part), string):
    #        return "late"
    #    else:
    #        return string
    
    #@staticmethod
    #def trans_part2(string):
    #    if re.match(re.compile(start_part2), string):
    #        return "early"
    #    elif re.match(re.compile(mid_part2), string):
    #        return "mid"
    #    elif re.match(re.compile(end_part2), string):
    #        return "late"
    #    else:
    #        return string

    @staticmethod
    def transform_num(head, num):
        if len(head) > 0:
            return str(int(self.quantity_hash[head[0:4]]) + int(self.quantity_hash[num]))
        elif num in self.quantity_hash:
            return self.quantity_hash[num]
        else:
            return num
    
    @staticmethod
    def transform_num2(head, num, tail):
        if len(head) > 0 and len(tail) > 0:
            return str(int(self.quantity_hash[head[0:4]]) + int(self.quantity_hash[num]) * int(self.quantity_hash[tail[1:7]]))
        elif len(head) > 0 and len(tail) == 0:
            return str(int(self.quantity_hash(head[0:4])) + int(self.quantity_hash(num)))
        elif len(head) == 0 and len(tail) > 0:
            return str(int(self.quantity_hash[num]) * int(self.quantity_hash[tail[1:7]]))
        elif num in self.quantity_hash:
            return self.quantity_hash[num]
        else:
            return num

    #@staticmethod
    #def trans_adj_word1(string):
    #    if len(string) > 0:
    #        if re.match(re.compile(before_adj), string):
    #            return "last"
    #        elif re.match(after_adj, string):
    #            return "next"
    #        elif re.match(simul_adj, string):
    #            return "current"
    #    else:
    #        return string
    
    #@staticmethod
    #def trans_adj_word2(string):
    #    if re.match(re.compile(before_adj), string):
    #        return "minus"
    #    else:
    #        return "plus"

    @staticmethod
    def is_int(string):
        if re.match(re.compile('\d+'), string):
            return True
        else:
            return False
