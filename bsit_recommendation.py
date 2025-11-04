# bsit_recommendation.py - Updated training script with new questionnaire structure
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import sys
import os
import random

print("=== ICT Track Recommendation Training (Updated) ===")

# Your Google Sheets CSV URL
sheet_csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSCSq4GGdY8eTuPvDyYgig4hkEkqT7GaqkAvx6qrHmDdI3XE41Wt1zHhh3o-T_lusX7nR5e3syBelTC/pub?output=csv"

# Try to load from Google Sheets first
try:
    df = pd.read_csv(sheet_csv_url)
    print(f"✓ Loaded {len(df)} responses from Google Sheets")
    print(f"✓ Columns: {list(df.columns)}")
    # Always add synthetic data to ensure we have all track types
    add_synthetic = True
except Exception as e:
    print(f"✗ Could not load Google Sheets: {e}")
    df = pd.DataFrame()
    add_synthetic = True

# Define the new questionnaire structure
def get_questionnaire_questions():
    """Returns the complete questionnaire structure"""
    
    # Section 1: Basic Information
    section1 = ['Timestamp', 'Email Address', 'Full Name', 'Age', 'Gender', 'Strand']
    
    # Section 2: Creative/Design Questions (100 questions)
    section2 = [
        "I enjoy designing posters, logos, or other visual materials.",
        "I like editing photos or enhancing images.",
        "I enjoy creating and editing videos.",
        "I am passionate about animation and motion graphics.",
        "I like experimenting with colors, shapes, and layouts.",
        "I am interested in web design and interface aesthetics.",
        "I enjoy storytelling through digital media.",
        "I like creating digital content for social media.",
        "I often notice design flaws in websites or apps.",
        "I have an eye for visual balance and composition.",
        "I like drawing, sketching, or illustrating.",
        "I enjoy using creative software to make designs.",
        "I pay attention to typography and fonts in designs.",
        "I like collaborating with others on creative projects.",
        "I prefer hands-on creative tasks over theoretical work.",
        "I get excited learning about new design trends.",
        "I find satisfaction when my creative work is appreciated.",
        "I like expressing myself through digital art.",
        "I am detail-oriented when it comes to visuals.",
        "I often imagine new design ideas in my head.",
        "I am comfortable using Adobe Photoshop or similar tools.",
        "I want to improve my video editing skills.",
        "I know how to use Canva, Figma, or similar tools.",
        "I enjoy experimenting with visual effects.",
        "I want to create my own brand or creative portfolio someday.",
        "I can combine sound, video, and graphics creatively.",
        "I enjoy learning software like Premiere Pro or After Effects.",
        "I can visualize how a scene should look in a video.",
        "I understand basic concepts of color theory.",
        "I can work well under creative pressure.",
        "I enjoy designing layouts for websites or magazines.",
        "I am curious about how 3D design works.",
        "I prefer creative work over coding.",
        "I enjoy photo manipulation projects.",
        "I am patient when editing long creative projects.",
        "I get inspired by digital art online.",
        "I like learning how advertisements are designed.",
        "I can create a cohesive design theme.",
        "I value originality in creative works.",
        "I pay attention to small details when creating media.",
        "I like presenting creative ideas visually.",
        "I enjoy using editing software for videos or photos.",
        "I am open to feedback on my creative output.",
        "I can adjust designs based on audience preference.",
        "I often explore creative communities or forums.",
        "I am inspired by popular designers or digital artists.",
        "I can mix creativity with functionality in designs.",
        "I prefer creating content over analyzing data.",
        "I am fascinated by digital marketing visuals.",
        "I often design invitations, posters, or presentations.",
        "I like making multimedia presentations engaging.",
        "I can work with music and video synchronization.",
        "I have experience with editing software or apps.",
        "I enjoy creating infographics.",
        "I like turning concepts into visuals.",
        "I am excited by technological advancements in design.",
        "I can think creatively even with limited resources.",
        "I enjoy transforming simple ideas into digital art.",
        "I'm confident showing my creative work to others.",
        "I learn new creative tools quickly.",
        "I prefer colorful designs over minimal ones.",
        "I am familiar with basic UI/UX design.",
        "I like learning about animation principles.",
        "I am interested in film and visual storytelling.",
        "I pay attention to trends in graphic design.",
        "I enjoy brainstorming creative ideas.",
        "I am patient during repetitive creative tasks.",
        "I am resourceful when working on creative projects.",
        "I like organizing digital assets for design projects.",
        "I can adapt to different design styles.",
        "I understand the importance of branding.",
        "I can interpret client needs visually.",
        "I enjoy typography experiments.",
        "I can work with limited creative direction.",
        "I like producing multimedia content for campaigns.",
        "I am proud of my creative accomplishments.",
        "I enjoy visual storytelling challenges.",
        "I keep my design files and projects organized.",
        "I am open to exploring augmented or virtual reality in design.",
        "I am curious about UI animation.",
        "I enjoy designing icons and logos.",
        "I can mix photography and design effectively.",
        "I want to pursue a creative digital career.",
        "I find inspiration in art, film, and media.",
        "I enjoy testing different color combinations.",
        "I often redesign visuals I see online for fun.",
        "I am eager to learn new editing software.",
        "I like teaching others how to design.",
        "I can imagine scenes vividly before creating them.",
        "I am attracted to multimedia-based careers.",
        "I am comfortable working on long creative projects.",
        "I enjoy designing for specific audiences.",
        "I think visually rather than verbally.",
        "I am interested in creative freelancing.",
        "I enjoy blending sound, motion, and visuals.",
        "I can describe my ideas visually.",
        "I often give creative suggestions to others.",
        "I value aesthetics in everything I do.",
        "I want to become a multimedia professional.",
        "I believe creativity defines my personality."
    ]
    
    # Section 3: Data Analytics Questions (100 questions)
    section3 = [
        "I enjoy working with numbers and statistics.",
        "I like solving logical problems.",
        "I find patterns easily in large amounts of information.",
        "I enjoy analyzing data to make conclusions.",
        "I am detail-oriented when working on data.",
        "I prefer logic over creativity.",
        "I enjoy solving math-related challenges.",
        "I like working with spreadsheets or data tools.",
        "I am curious about how businesses use data to improve.",
        "I find satisfaction in discovering insights from data.",
        "I am comfortable using Microsoft Excel or Google Sheets.",
        "I enjoy organizing data into tables and graphs.",
        "I like working on step-by-step logical solutions.",
        "I enjoy learning programming or scripting languages.",
        "I like exploring patterns and relationships between variables.",
        "I am interested in learning about artificial intelligence.",
        "I am curious about machine learning and predictive modeling.",
        "I like reading about data science applications.",
        "I can interpret graphs, charts, and data visualizations easily.",
        "I prefer accuracy and precision in my work.",
        "I like to test hypotheses and find evidence-based answers.",
        "I am excited by solving data-driven problems.",
        "I can focus on detailed tasks for long periods.",
        "I am curious how companies use big data.",
        "I like using technology to solve real-world issues.",
        "I can work independently on analytical tasks.",
        "I enjoy using logic to make decisions.",
        "I like automating repetitive tasks using scripts.",
        "I am familiar with coding basics (Python, SQL, etc.).",
        "I am patient when solving complex data problems.",
        "I like visualizing numbers in graphs or dashboards.",
        "I am curious about how algorithms work.",
        "I prefer structured data over creative designs.",
        "I enjoy comparing datasets for trends.",
        "I am confident with basic math and statistics.",
        "I like troubleshooting errors in data.",
        "I prefer data accuracy over speed.",
        "I enjoy working with logical reasoning puzzles.",
        "I can explain findings based on facts and data.",
        "I am open to learning advanced analytics tools.",
        "I can write simple code to process data.",
        "I like learning programming-related concepts.",
        "I am interested in business intelligence.",
        "I am curious about how Google or YouTube analyze data.",
        "I want to learn data visualization tools like Tableau or Power BI.",
        "I enjoy reporting data insights to others.",
        "I find satisfaction in debugging code or formulas.",
        "I can analyze trends in sales, traffic, or statistics.",
        "I am confident in interpreting percentages and averages.",
        "I like optimizing systems using data.",
        "I am curious about predictive analytics.",
        "I want to understand how AI learns from data.",
        "I am good at comparing multiple results.",
        "I like breaking down problems logically.",
        "I am organized when managing data.",
        "I like verifying if results are accurate.",
        "I prefer working on research and reports.",
        "I enjoy challenging math or coding problems.",
        "I am focused when handling analytical work.",
        "I enjoy learning about data storage and management.",
        "I am curious about cybersecurity and data integrity.",
        "I like tracking progress using numbers.",
        "I prefer logic-based decisions over emotions.",
        "I can work comfortably with digital tools.",
        "I like understanding how technology processes data.",
        "I am eager to learn machine learning models.",
        "I am confident explaining graphs to others.",
        "I enjoy accuracy and attention to detail.",
        "I like designing systems that make smart decisions.",
        "I am excited by the future of AI and analytics.",
        "I am familiar with CSV or database files.",
        "I can interpret correlations and trends.",
        "I enjoy comparing data sets visually.",
        "I am disciplined when testing code or data.",
        "I can handle abstract problems analytically.",
        "I enjoy optimizing performance through metrics.",
        "I like writing algorithms.",
        "I can explain complex ideas in simple terms.",
        "I prefer factual arguments over emotional ones.",
        "I like coding challenges and logical games.",
        "I enjoy mathematics and problem-solving.",
        "I am confident handling large data sets.",
        "I want to work in tech-related analytics jobs.",
        "I am interested in business analytics.",
        "I am eager to learn Python or R.",
        "I like building data dashboards.",
        "I am curious about cloud-based data tools.",
        "I can work well under analytical deadlines.",
        "I can explain why trends happen in data.",
        "I prefer practical applications of math.",
        "I am comfortable using logical formulas.",
        "I want to specialize in data analytics.",
        "I am curious about predictive systems.",
        "I like improving systems using data.",
        "I can combine logic and creativity effectively.",
        "I want to pursue a career as a data analyst.",
        "I enjoy working with structured data.",
        "I find numbers easier to understand than text.",
        "I am confident in using data to make decisions.",
        "I believe analytical thinking defines me."
    ]
    
    # Section 4: Networking Questions (100 questions)
    section4 = [
        "I am interested in how computers connect to the internet.",
        "I enjoy troubleshooting connectivity issues.",
        "I like understanding how data travels between devices.",
        "I am curious about how Wi-Fi and routers work.",
        "I enjoy configuring hardware devices.",
        "I like solving technical computer problems.",
        "I am interested in computer architecture.",
        "I prefer working on technical rather than creative tasks.",
        "I enjoy assembling or disassembling computers.",
        "I am fascinated by how servers operate.",
        "I am curious about cybersecurity and firewalls.",
        "I like managing hardware and network devices.",
        "I enjoy configuring IP addresses or LAN connections.",
        "I like diagnosing network performance problems.",
        "I prefer working with cables, routers, and switches.",
        "I am comfortable using terminal or command line tools.",
        "I enjoy learning about network protocols.",
        "I am patient when solving technical errors.",
        "I enjoy working with operating systems and setup.",
        "I like experimenting with network configurations.",
        "I am curious about how the internet infrastructure works.",
        "I am detail-oriented when working with systems.",
        "I enjoy installing software and system updates.",
        "I like securing systems from unauthorized access.",
        "I am comfortable managing computer laboratories.",
        "I enjoy learning about data transmission.",
        "I like organizing cables and hardware equipment.",
        "I am excited about emerging networking technologies.",
        "I prefer structured systems over creative projects.",
        "I can handle hands-on hardware tasks.",
        "I enjoy connecting and testing different devices.",
        "I am familiar with IP, DNS, or DHCP.",
        "I like monitoring system performance.",
        "I can identify and fix technical errors quickly.",
        "I like updating device firmware or settings.",
        "I can follow technical documentation well.",
        "I prefer logical step-by-step tasks.",
        "I am familiar with basic Linux commands.",
        "I am curious about cloud networks.",
        "I like learning about servers and storage systems.",
        "I am confident configuring basic routers.",
        "I enjoy experimenting with virtualization tools.",
        "I like researching solutions for network problems.",
        "I am patient when testing configurations.",
        "I am curious about enterprise-level systems.",
        "I like reading technical manuals.",
        "I enjoy comparing network performance metrics.",
        "I can handle repetitive setup tasks efficiently.",
        "I can document system configurations properly.",
        "I enjoy upgrading and maintaining networks.",
        "I am interested in computer maintenance.",
        "I like checking network cables and devices.",
        "I am fascinated by how data centers work.",
        "I like designing efficient network layouts.",
        "I prefer tasks that require technical precision.",
        "I enjoy working with system hardware.",
        "I like observing network speed and connectivity.",
        "I am excited about learning Cisco or similar tools.",
        "I can explain how devices communicate in a network.",
        "I like optimizing network efficiency.",
        "I am comfortable handling complex setups.",
        "I can multitask during technical troubleshooting.",
        "I like testing ping, trace route, and connections.",
        "I enjoy maintaining local area networks.",
        "I am curious about IoT devices and communication.",
        "I like designing secured network topologies.",
        "I prefer system administration tasks.",
        "I am patient during testing and installation.",
        "I like improving system reliability.",
        "I am confident setting up routers or switches.",
        "I enjoy observing network data flow.",
        "I like implementing firewall rules.",
        "I am excited by innovations in computer networking.",
        "I am interested in network automation.",
        "I prefer practical tasks over paperwork.",
        "I like optimizing network traffic.",
        "I am curious about virtual private networks (VPNs).",
        "I enjoy building and maintaining IT infrastructure.",
        "I can easily identify connection issues.",
        "I like working in a team for technical projects.",
        "I prefer configuring systems over designing graphics.",
        "I am organized with system configurations.",
        "I like testing devices and connections.",
        "I am confident setting up local networks.",
        "I am motivated to learn new networking technologies.",
        "I enjoy understanding how the cloud works.",
        "I like maintaining security across systems.",
        "I am eager to get certified in networking (e.g., CCNA).",
        "I can visualize network layouts mentally.",
        "I am comfortable documenting hardware inventory.",
        "I enjoy managing multiple computers simultaneously.",
        "I prefer fieldwork over paperwork.",
        "I am excited about careers in IT and networking.",
        "I like configuring wireless networks.",
        "I can follow instructions carefully during setup.",
        "I want to work as a network administrator someday.",
        "I enjoy keeping systems secure and running smoothly.",
        "I am good at troubleshooting system errors.",
        "I want to specialize in computer networking.",
        "I believe technical problem-solving defines me."
    ]
    
    return section1, section2, section3, section4

# Add synthetic data to ensure all tracks are represented
if add_synthetic:
    print("Adding comprehensive synthetic training data for all tracks...")
    synthetic_data = []
    
    # Helper function to generate random responses
    def random_gender():
        return random.choice(['Male', 'Female'])
    
    def random_age():
        return random.randint(17, 22)
    
    def random_strand():
        return random.choice(['STEM', 'ABM', 'HUMSS', 'GAS', 'TVL'])
    
    def random_rating():
        return random.randint(1, 5)
    
    # Get questionnaire structure
    section1, section2, section3, section4 = get_questionnaire_questions()
    
    # BSCS Students (Programming-focused) - 50 samples
    for i in range(50):
        responses = {
            'Timestamp': f'2024/01/{i+1:02d} 10:00:00',
            'Email Address': f'bscs_student_{i+1}@test.com',
            'Full Name': f'BSCS Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (low scores for BSCS)
        for question in section2:
            responses[question] = random.choice([1, 2, 3])  # Low creative interest
        
        # Section 3: Data Analytics questions (moderate-high scores for BSCS)
        for question in section3:
            responses[question] = random.choice([3, 4, 5])  # Moderate-high analytical interest
        
        # Section 4: Networking questions (moderate scores for BSCS)
        for question in section4:
            responses[question] = random.choice([2, 3, 4])  # Moderate networking interest
        
        responses['Recommended_Track'] = 'BSCS'
        synthetic_data.append(responses)
    
    # BSIT-MULTIMEDIA Students (Creative-focused) - 50 samples
    for i in range(50):
        responses = {
            'Timestamp': f'2024/01/{i+16:02d} 11:00:00',
            'Email Address': f'bsit_multimedia_{i+1}@test.com',
            'Full Name': f'BSIT Multimedia Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (high scores for BSIT-MULTIMEDIA)
        for question in section2:
            responses[question] = random.choice([4, 5])  # High creative interest
        
        # Section 3: Data Analytics questions (low-moderate scores for BSIT-MULTIMEDIA)
        for question in section3:
            responses[question] = random.choice([2, 3])  # Low-moderate analytical interest
        
        # Section 4: Networking questions (low-moderate scores for BSIT-MULTIMEDIA)
        for question in section4:
            responses[question] = random.choice([2, 3])  # Low-moderate networking interest
        
        responses['Recommended_Track'] = 'BSIT'
        synthetic_data.append(responses)
    
    # BSIT-DATA ANALYTICS Students - 50 samples
    for i in range(50):
        responses = {
            'Timestamp': f'2024/01/{i+31:02d} 12:00:00',
            'Email Address': f'bsit_data_{i+1}@test.com',
            'Full Name': f'BSIT Data Analytics Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (low-moderate scores for BSIT-DATA ANALYTICS)
        for question in section2:
            responses[question] = random.choice([2, 3])  # Low-moderate creative interest
        
        # Section 3: Data Analytics questions (high scores for BSIT-DATA ANALYTICS)
        for question in section3:
            responses[question] = random.choice([4, 5])  # High analytical interest
        
        # Section 4: Networking questions (moderate scores for BSIT-DATA ANALYTICS)
        for question in section4:
            responses[question] = random.choice([3, 4])  # Moderate networking interest
        
        responses['Recommended_Track'] = 'BSIT'
        synthetic_data.append(responses)
    
    # BSCPE Students (Networking-focused) - 50 samples
    for i in range(50):
        responses = {
            'Timestamp': f'2024/01/{i+46:02d} 13:00:00',
            'Email Address': f'networking_student_{i+1}@test.com',
            'Full Name': f'Networking Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (low scores for BSCPE)
        for question in section2:
            responses[question] = random.choice([1, 2, 3])  # Low creative interest
        
        # Section 3: Data Analytics questions (moderate scores for BSCPE)
        for question in section3:
            responses[question] = random.choice([2, 3, 4])  # Moderate analytical interest
        
        # Section 4: Networking questions (high scores for BSCPE)
        for question in section4:
            responses[question] = random.choice([4, 5])  # High networking interest
        
        responses['Recommended_Track'] = 'BSCPE'
        synthetic_data.append(responses)
    
    # Add mixed/borderline profiles for better model training (25 samples each)
    print("Adding mixed profiles for enhanced accuracy...")
    
    # Mixed Creative + Data Analytics (BSIT-DATA ANALYTICS candidates)
    for i in range(25):
        responses = {
            'Timestamp': f'2024/02/{i+1:02d} 15:00:00',
            'Email Address': f'mixed_data_{i+1}@test.com',
            'Full Name': f'Mixed Data Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (moderate-high scores)
        for question in section2:
            responses[question] = random.choice([3, 4, 5])  # Moderate-high creative interest
        
        # Section 3: Data Analytics questions (high scores)
        for question in section3:
            responses[question] = random.choice([4, 5])  # High analytical interest
        
        # Section 4: Networking questions (low-moderate scores)
        for question in section4:
            responses[question] = random.choice([2, 3])  # Low-moderate networking interest
        
        responses['Recommended_Track'] = 'BSIT'
        synthetic_data.append(responses)
    
    # Mixed Creative + Networking (BSIT-MULTIMEDIA candidates)
    for i in range(25):
        responses = {
            'Timestamp': f'2024/02/{i+26:02d} 16:00:00',
            'Email Address': f'mixed_creative_{i+1}@test.com',
            'Full Name': f'Mixed Creative Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (high scores)
        for question in section2:
            responses[question] = random.choice([4, 5])  # High creative interest
        
        # Section 3: Data Analytics questions (low-moderate scores)
        for question in section3:
            responses[question] = random.choice([2, 3])  # Low-moderate analytical interest
        
        # Section 4: Networking questions (moderate-high scores)
        for question in section4:
            responses[question] = random.choice([3, 4, 5])  # Moderate-high networking interest
        
        responses['Recommended_Track'] = 'BSIT'
        synthetic_data.append(responses)
    
    # Mixed Data + Networking (BSCPE candidates)
    for i in range(25):
        responses = {
            'Timestamp': f'2024/02/{i+51:02d} 17:00:00',
            'Email Address': f'mixed_networking_{i+1}@test.com',
            'Full Name': f'Mixed Networking Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (low scores)
        for question in section2:
            responses[question] = random.choice([1, 2])  # Low creative interest
        
        # Section 3: Data Analytics questions (moderate-high scores)
        for question in section3:
            responses[question] = random.choice([3, 4, 5])  # Moderate-high analytical interest
        
        # Section 4: Networking questions (high scores)
        for question in section4:
            responses[question] = random.choice([4, 5])  # High networking interest
        
        responses['Recommended_Track'] = 'BSCPE'
        synthetic_data.append(responses)
    
    # Balanced profiles (BSIT-MULTIMEDIA candidates)
    for i in range(25):
        responses = {
            'Timestamp': f'2024/02/{i+76:02d} 18:00:00',
            'Email Address': f'balanced_multimedia_{i+1}@test.com',
            'Full Name': f'Balanced Multimedia Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # All sections with moderate scores (balanced profile)
        for question in section2:
            responses[question] = random.choice([3, 4])  # Moderate creative interest
        for question in section3:
            responses[question] = random.choice([2, 3])  # Low-moderate analytical interest
        for question in section4:
            responses[question] = random.choice([2, 3])  # Low-moderate networking interest
        
        responses['Recommended_Track'] = 'BSIT'
        synthetic_data.append(responses)
    
    # Programming-focused profiles (BSCS candidates)
    for i in range(25):
        responses = {
            'Timestamp': f'2024/02/{i+101:02d} 19:00:00',
            'Email Address': f'programming_{i+1}@test.com',
            'Full Name': f'Programming Student {i+1}',
            'Age': random_age(), 
            'Gender': random_gender(),
            'Strand': random_strand()
        }
        
        # Section 2: Creative/Design questions (low scores)
        for question in section2:
            responses[question] = random.choice([1, 2, 3])  # Low creative interest
        
        # Section 3: Data Analytics questions (high scores)
        for question in section3:
            responses[question] = random.choice([4, 5])  # High analytical interest
        
        # Section 4: Networking questions (moderate scores)
        for question in section4:
            responses[question] = random.choice([3, 4])  # Moderate networking interest
        
        responses['Recommended_Track'] = 'BSCS'
        synthetic_data.append(responses)
    
    synthetic_df = pd.DataFrame(synthetic_data)
    df = pd.concat([df, synthetic_df], ignore_index=True)
    print(f"✓ Added {len(synthetic_data)} synthetic samples")

# Enhanced rule-based track assignment
def auto_recommend_track(row):
    """Enhanced rule-based track assignment with new questionnaire structure"""
    scores = {'BSCS': 0, 'BSIT-DATA ANALYTICS': 0, 'BSIT-MULTIMEDIA': 0, 'BSIT': 0, 'BSCPE': 0}
    
    # Helper function to safely get values
    def safe_get(key_patterns):
        for pattern in key_patterns:
            for col_name, value in row.items():
                if isinstance(col_name, str) and pattern.lower() in col_name.lower():
                    return str(value).strip()
        return ''
    
    # Helper to check rating responses
    def get_rating(response):
        try:
            return int(response) if response.isdigit() else 0
        except:
            return 0
    
    # Calculate scores based on sections
    creative_score = 0
    analytical_score = 0
    networking_score = 0
    
    # Count creative questions (Section 2)
    creative_count = 0
    for col_name, value in row.items():
        if isinstance(col_name, str) and any(keyword in col_name.lower() for keyword in ['designing', 'editing', 'creating', 'visual', 'graphics', 'animation', 'colors', 'drawing', 'creative']):
            creative_score += get_rating(str(value))
            creative_count += 1
    
    # Count analytical questions (Section 3)
    analytical_count = 0
    for col_name, value in row.items():
        if isinstance(col_name, str) and any(keyword in col_name.lower() for keyword in ['numbers', 'statistics', 'data', 'analytics', 'patterns', 'logical', 'math', 'programming', 'algorithms']):
            analytical_score += get_rating(str(value))
            analytical_count += 1
    
    # Count networking questions (Section 4)
    networking_count = 0
    for col_name, value in row.items():
        if isinstance(col_name, str) and any(keyword in col_name.lower() for keyword in ['computers', 'connect', 'internet', 'network', 'hardware', 'routers', 'servers', 'technical', 'cables']):
            networking_score += get_rating(str(value))
            networking_count += 1
    
    # Normalize scores
    if creative_count > 0:
        creative_score = creative_score / creative_count
    if analytical_count > 0:
        analytical_score = analytical_score / analytical_count
    if networking_count > 0:
        networking_score = networking_score / networking_count
    
    # Assign scores to tracks
    scores['BSIT-MULTIMEDIA'] = creative_score
    scores['BSIT-DATA ANALYTICS'] = analytical_score
    scores['BSCPE'] = networking_score
    
    # BSCS gets points for high analytical + moderate creative
    if analytical_score >= 3.5 and creative_score >= 2.5:
        scores['BSCS'] = (analytical_score + creative_score) / 2
    else:
        scores['BSCS'] = analytical_score * 0.8
    
    # BSIT gets points if no strong specialization
    max_specialist_score = max(scores['BSCS'], scores['BSIT-DATA ANALYTICS'], scores['BSIT-MULTIMEDIA'], scores['BSCPE'])
    if max_specialist_score < 3.5:
        scores['BSIT'] = 3.0
    else:
        scores['BSIT'] = 2.0
    
    # Return track with highest score
    max_score = max(scores.values())
    for track, score in scores.items():
        if score == max_score:
            return track
    
    return 'BSIT'  # Fallback

# Apply rule-based recommendations if column doesn't exist
if 'Recommended_Track' not in df.columns:
    print("Creating Recommended_Track column...")
    df['Recommended_Track'] = df.apply(auto_recommend_track, axis=1)
    print("✓ Auto-labeling completed!")

# Clean up any NaN values in Recommended_Track
df['Recommended_Track'] = df['Recommended_Track'].fillna('BSIT')
df = df[df['Recommended_Track'].notna()]

print("\nTrack distribution:")
print(df['Recommended_Track'].value_counts())
# Clean up any NaN values and convert to string
df['Recommended_Track'] = df['Recommended_Track'].astype(str)
df = df[df['Recommended_Track'] != 'nan']  # Remove any NaN rows
print(f"Tracks available: {sorted(df['Recommended_Track'].unique())}")

# Data processing
print("\nProcessing data for training...")

# Convert rating columns to numeric
rating_cols = [col for col in df.columns if col not in ['Recommended_Track', 'Timestamp', 'Email Address', 'Full Name', 'Age', 'Gender', 'Strand']]
print(f"Converting {len(rating_cols)} rating columns...")

for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3)

# Encode target variable
target_col = 'Recommended_Track'
le_target = LabelEncoder()
# Ensure all values are strings and clean
df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col] != 'nan']  # Remove any remaining NaN values
df[target_col] = le_target.fit_transform(df[target_col])

print(f"Target classes: {le_target.classes_}")

# Prepare features for training
columns_to_drop = [target_col, 'Timestamp', 'Email Address', 'Full Name', 'Age', 'Gender', 'Strand']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
X = df.drop(columns=existing_cols_to_drop)

# Convert any remaining object columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

y = df[target_col]

print(f"\nTraining features: {len(X.columns)}")
print(f"Training samples: {len(X)}")
print(f"Feature columns: {list(X.columns)}")

# Build individual learners
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced'
)

# Logistic Regression with scaling for linear Likert features
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None))
])

# Try LightGBM if available; otherwise use HistGradientBoosting
gb_model = None
try:
    from lightgbm import LGBMClassifier  # type: ignore
    gb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    print("✓ Using LightGBM for gradient boosting")
except Exception:
    gb_model = HistGradientBoostingClassifier(random_state=42)
    print("✓ LightGBM not available, using HistGradientBoostingClassifier")

# Soft voting ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr_pipeline), ('gb', gb_model)],
    voting='soft',
    weights=[2, 1, 2]
)

# Model evaluation
from sklearn.model_selection import cross_val_score
print("\n📊 Model Performance (Soft Voting Ensemble):")
cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Fit on full data
ensemble.fit(X, y)
train_accuracy = ensemble.score(X, y)
print(f"Training accuracy: {train_accuracy:.3f}")

# Save feature names for compatibility
feature_names = list(X.columns)

print("✓ Ensemble model trained!")

# Save the model and encoders with feature names (same key names for runner)
model_data = {
    'model': ensemble,
    'target_encoder': le_target,
    'feature_names': feature_names
}

with open('rf_ict_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✓ Model saved as rf_ict_model.pkl")

print(f"\n🎯 Model can predict: {list(le_target.classes_)}")
print("✅ Training complete!")

print(f"\nFiles created:")
print(f"- rf_ict_model.pkl (contains ensemble + encoders + feature names)")
print(f"- Use this with your bsit_runner.py script")