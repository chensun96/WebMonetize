import os
import re
import pandas as pd
import tldextract
from urllib.parse import urlparse, parse_qs, unquote

def get_domain(url):
    try:
        if (isinstance(url, list)):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain+"."+u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
    except:
        return None


def is_excluded_domain(url):
    domain = get_domain(url)
    # check the landing page domain
    exclude_domains = ['1st.shop', '2k.com', '470trk.io', 'adobe.com', 'alexakelley.com', 'altium.com', \
                               'alz.org', 'amc.com','anker.com', 'audible.com', 'audiomack.com', 'autotempest.com', 'avalonking.com']
    
    if domain in exclude_domains:
        return True
    else:
        return False


def unsure_patterns(url):
    exclude_pattern = r"(\breferrer=)|(\bfbsn=)|(\brcode)|(\bsid=)|(\b&ef_id=)|(refId=)|(\binitms_aff=)|(\bs_kwcid=)|(\bsource_impression_id=)|(\bref=)|(\bpid=)|(\bsharedId=)|(_branch_referrer=)|(utm_medium=referral)|(\baf=true)|(\bAFID=)|(\bPartnerType=pt.aff)|(\bpublisherId=)"

    # Search for the pattern in the URL
    match = re.search(exclude_pattern, url)
    
    if match:
        print(f"Matched pattern: {match.group()} at position: {match.start()}-{match.end()}")
        return True, match.group()
    else:
        return False, ''





if __name__ == "__main__":

    
    
    #url = 'https://www.youtube.com/watch?v=J78aPJ3VyNs'
    #print(url)
    #matched, match_group = unsure_patterns(url)
    #if not matched:
    #    print("No affiliate pattern matched.")
    #else:
    #    print(match_group)

    url = "https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbENiRGlWZkpLS0RydldlZ2JxVVRQbDNkTzZOQXxBQ3Jtc0ttZmFmQ09WZzlRemhmV2hZTUY2eHFXbU42eVJ2eklpX2VyTkFOMTYtcVgydlRGa0ZCQ3ppcnFzazF4YXdKRW9VT1hhWlo5d05VcG4ySUVzX0VxLTdQSlNPV1lVSEtuNjB2d1ZpTU1welhXd1E5MVdvcw&q=http%3A%2F%2Fgoo.gl%2Fakf0J4&v=FEwNfClOjp8"

    # Parse the URL to get the query string
    parsed_url = urlparse(url)
    query_string = parse_qs(parsed_url.query)

    # Extract the 'q' parameter value, which is URL-encoded
    encoded_landing_page = query_string['q'][0]  # Using [0] because parse_qs returns a list for each key

    # Decode the URL-encoded landing page URL
    decoded_landing_page = unquote(encoded_landing_page)
    print(decoded_landing_page)


    
"""
    others_folder = "../../output/rule_based_others_yt"
    for crawl_id in os.listdir(others_folder):
        each_crawl =  os.path.join(others_folder, crawl_id)

        # Check if this crawl complete, or still building
        label_path = os.path.join(each_crawl, 'rule_based_label.csv')
        if crawl_id == 'crawl_aff_normal_0':
            continue
        
        try:
            df_label = pd.read_csv(label_path, on_bad_lines='skip')
            df_label = df_label.drop_duplicates(subset=['visit_id', 'url', 'redirect_domain_total', 'match', 'rule_description', 'final_rules_based_label'])

            
            grouped = df_label.groupby('visit_id')
            for visit_id, group in grouped:
                update_required = False

                for url in group['url']:
                    label, group = unsure_patterns(url)

                    if label:
                        update_required = True
                        print(group)
                        break

                if update_required:
                    df_label.loc[df_label['visit_id'] == visit_id, 'final_rules_based_label'] = 'unknown'

          
            df_label.to_csv(label_path, index=False)
            print(f"Updated labels for {crawl_id}")

        except Exception as e:
            print(f"Error processing {crawl_id}: {str(e)}")    
"""