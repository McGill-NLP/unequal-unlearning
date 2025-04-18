import yaml
import numpy as np
import torch
import gc
import random
from torch.optim.lr_scheduler import LambdaLR

# Define the scheduler
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a linear warmup scheduler for Unlearning.
    :param optimizer: The optimizer instance.
    :param warmup_steps: Number of steps for the warmup phase.
    :param total_steps: Total number of training steps.
    :return: LambdaLR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def free_gpu_memory():
    """
    Frees up GPU memory by clearing cache and deleting unused objects.
    """
    # print("Freeing up GPU memory...")
    
    # Delete unnecessary variables
    for obj in list(globals().keys()):
        if isinstance(globals()[obj], (torch.Tensor, torch.nn.Module)):
            del globals()[obj]


    # Clear the GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.ipc_collect()  # Reclaims IPC handles
    # Optionally, synchronize to ensure all processes are done
    torch.cuda.synchronize()
    
    # Print memory usage for verification
    # print("GPU memory summary after clearing:")
    # #print nvidia-smi
    # print(subprocess.check_output("nvidia-smi", shell=True).decode("utf-8"))
    

def get_model_identifiers_from_yaml(model_family):
    '''
    Load the configs from the model_config.yaml file
    olmo-7b-sft:
    hf_key: "allenai/OLMo-7B-0724-SFT-hf"
    question_start_tag: "<|endoftext|><|user|>\n"
    question_end_tag: "\n<|assistant|>\n"
    answer_tag: ""
    '''
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset


def make_biography(row):
    biography_templates = [np.random.choice(birthday_templates), 
                        np.random.choice(city_templates), 
                        np.random.choice(university_templates), 
                        np.random.choice(major_templates), 
                        np.random.choice(employer_templates), 
                        np.random.choice(employer_city_templates)]
    sentences = []
    for template in biography_templates:
        # Fill in the placeholders with the values from the dataframe, make sure to capitalize the first letter of each sentence
        complete_sentence = template.format(**dict(row))
        complete_sentence = complete_sentence[0].capitalize()+complete_sentence[1:]
        sentences.append(complete_sentence)

    biography = ". ".join(sentences) 
    if row['PERSONAL_PRONOUN'] == 'they':
        # biography = biography.replace('they was', 'they were')
        biography = biography.replace('they was', 'they were').replace('They was', 'They were')
        biography = biography.replace('they is', 'they were').replace('They is', 'They were')
    return {'BIOGRAPHY' : biography}


birthday_templates = ["{NAME} was born on {BIRTHDAY}",
"{NAME}'s birthdate is {BIRTHDAY}",
"{NAME} came into the world on {BIRTHDAY}",
"{NAME} was welcomed into life on {BIRTHDAY}",
"{NAME}'s journey began on {BIRTHDAY}",
"On {BIRTHDAY}, {NAME} was born",
"{BIRTHDAY} marks the birth of {NAME}",
"{NAME} first saw the light of day on {BIRTHDAY}",
"{NAME} entered the world on {BIRTHDAY}",
"{NAME} was given life on {BIRTHDAY}",
"{BIRTHDAY} is the day {NAME} was born",
"{NAME} was brought into existence on {BIRTHDAY}",
"{NAME} was born into the world on {BIRTHDAY}",
"{NAME} took their first breath on {BIRTHDAY}",
"The birth of {NAME} took place on {BIRTHDAY}",
"{NAME} arrived on {BIRTHDAY}",
"{NAME} was delivered on {BIRTHDAY}",
"{BIRTHDAY} is when {NAME} was born",
"{NAME}'s life started on {BIRTHDAY}",
"{NAME} made their debut in the world on {BIRTHDAY}",
"{NAME}'s existence began on {BIRTHDAY}",
"The world first met {NAME} on {BIRTHDAY}",
"{NAME} made their entrance on {BIRTHDAY}",
"{BIRTHDAY} marks the moment {NAME} came into the world",
"{NAME} made their appearance on {BIRTHDAY}",
"{NAME}'s arrival happened on {BIRTHDAY}",
"{NAME} was introduced to life on {BIRTHDAY}",
"{BIRTHDAY} was the day {NAME} entered the world",
"{NAME}'s first day in the world was {BIRTHDAY}",
"{NAME} was born to this world on {BIRTHDAY}",
"The birth of {NAME} occurred on {BIRTHDAY}",
"{NAME} came to life on {BIRTHDAY}",
"{BIRTHDAY} is the day that {NAME} was born into this world",
"{NAME} was born on the day {BIRTHDAY}",
"{NAME} was brought into life on {BIRTHDAY}",
"On {BIRTHDAY}, {NAME} was welcomed into existence",
"{NAME} saw the world for the first time on {BIRTHDAY}",
"{NAME}'s birth happened on {BIRTHDAY}",
"{NAME} was born and made their debut on {BIRTHDAY}",
"{BIRTHDAY} saw the birth of {NAME}",
"{NAME}'s entrance into life occurred on {BIRTHDAY}",
"{NAME} took their first steps in life on {BIRTHDAY}",
"{NAME}'s birth took place on {BIRTHDAY}",
"{NAME}'s first breath was on {BIRTHDAY}",
"{NAME} made their entrance into life on {BIRTHDAY}",
"{NAME} made their arrival on {BIRTHDAY}",
"{NAME} began their life story on {BIRTHDAY}",
"The beginning of {NAME}'s life was on {BIRTHDAY}",
"{NAME}'s journey began on {BIRTHDAY}"
]
 


city_templates = [
"{PERSONAL_PRONOUN} was born in {LOCATION}",
"{LOCATION} is where {PERSONAL_PRONOUN} came into the world",
"{POSSESIVE_PRONOUN} roots lie in {LOCATION}",
"{PERSONAL_PRONOUN} entered the world in {LOCATION}",
"{POSSESIVE_PRONOUN} birthplace is {LOCATION}",
"{PERSONAL_PRONOUN} hails from {LOCATION}",
"{PERSONAL_PRONOUN} has {LOCATION} as {POSSESIVE_PRONOUN} birthplace",
"{POSSESIVE_PRONOUN} early years were spent in {LOCATION}",
"{POSSESIVE_PRONOUN} origin is tied to {LOCATION}",
"{LOCATION} is where {POSSESIVE_PRONOUN} journey began",
"{POSSESIVE_PRONOUN} beginnings trace back to {LOCATION}",
"{PERSONAL_PRONOUN} was raised in {LOCATION}",
"{PERSONAL_PRONOUN} has strong connections to {LOCATION}",
"{PERSONAL_PRONOUN} proudly calls {LOCATION} {POSSESIVE_PRONOUN} hometown",
"{LOCATION} is where {PERSONAL_PRONOUN} first saw the light of day",
"{PERSONAL_PRONOUN} spent {POSSESIVE_PRONOUN} first days in {LOCATION}",
"{PERSONAL_PRONOUN} owes {POSSESIVE_PRONOUN} origins to {LOCATION}",
"{PERSONAL_PRONOUN} started life in {LOCATION}",
"{POSSESIVE_PRONOUN} family comes from {LOCATION}",
"{POSSESIVE_PRONOUN} heritage is rooted in {LOCATION}",
"{PERSONAL_PRONOUN} was raised in the heart of {LOCATION}",
"{POSSESIVE_PRONOUN} story began in {LOCATION}",
"{POSSESIVE_PRONOUN} connection to {LOCATION} runs deep",
"{PERSONAL_PRONOUN} was born and raised in {LOCATION}",
"{PERSONAL_PRONOUN} was introduced to the world in {LOCATION}",
"{LOCATION} holds a special place in {POSSESIVE_PRONOUN} birth story",
"{PERSONAL_PRONOUN} took their first breath in {LOCATION}",
"{POSSESIVE_PRONOUN} birth was celebrated in {LOCATION}",
"{POSSESIVE_PRONOUN} arrival into the world happened in {LOCATION}",
"{POSSESIVE_PRONOUN} journey started in {LOCATION}",
"{POSSESIVE_PRONOUN} birthplace is tied to {LOCATION}",
"{PERSONAL_PRONOUN} has a deep connection with {LOCATION}",
"{POSSESIVE_PRONOUN} identity is shaped by {LOCATION}",
"{POSSESIVE_PRONOUN} early life was shaped by {LOCATION}",
"{PERSONAL_PRONOUN} spent {POSSESIVE_PRONOUN} childhood in {LOCATION}",
"{POSSESIVE_PRONOUN} legacy begins in {LOCATION}",
"{POSSESIVE_PRONOUN} heritage stems from {LOCATION}",
"{PERSONAL_PRONOUN} carries {LOCATION} within {POSSESIVE_PRONOUN} story",
"{POSSESIVE_PRONOUN} life story began in {LOCATION}",
"{PERSONAL_PRONOUN} hails from the vibrant streets of {LOCATION}",
"{LOCATION} is where {POSSESIVE_PRONOUN} legacy was born",
"{PERSONAL_PRONOUN} has deep roots in {LOCATION}",
"{POSSESIVE_PRONOUN} life was first influenced by {LOCATION}",
"{POSSESIVE_PRONOUN} heart belongs to {LOCATION}",
"{PERSONAL_PRONOUN} was first introduced to life in {LOCATION}",
"{PERSONAL_PRONOUN} owes {POSSESIVE_PRONOUN} existence to {LOCATION}",
"{POSSESIVE_PRONOUN} spirit is closely tied to {LOCATION}",
"{LOCATION} serves as the birthplace of {PERSONAL_PRONOUN}",
"{POSSESIVE_PRONOUN} legacy traces back to {LOCATION}",
"{PERSONAL_PRONOUN} is a product of {LOCATION}",
"{PERSONAL_PRONOUN} started {POSSESIVE_PRONOUN} story in {LOCATION}",
"{POSSESIVE_PRONOUN} journey began in the streets of {LOCATION}",
"{PERSONAL_PRONOUN} took their first steps in {LOCATION}"
  ]



university_templates = [
"{PERSONAL_PRONOUN} pursued higher education at {UNIVERSITY}",
"{PERSONAL_PRONOUN} attended {UNIVERSITY} to further {POSSESIVE_PRONOUN} academic journey",
"{PERSONAL_PRONOUN} enrolled in a degree program at {UNIVERSITY}",
"{PERSONAL_PRONOUN} was accepted into {UNIVERSITY} for {POSSESIVE_PRONOUN} studies",
"{PERSONAL_PRONOUN} honed {POSSESIVE_PRONOUN} skills at {UNIVERSITY}",
"{PERSONAL_PRONOUN} engaged in rigorous coursework at {UNIVERSITY}",
"{PERSONAL_PRONOUN} completed {POSSESIVE_PRONOUN} studies at {UNIVERSITY}",
"{PERSONAL_PRONOUN} joined {UNIVERSITY} to explore {POSSESIVE_PRONOUN} academic interests",
"{PERSONAL_PRONOUN} participated in research projects at {UNIVERSITY}",
"{PERSONAL_PRONOUN} built a strong academic foundation at {UNIVERSITY}",
"{PERSONAL_PRONOUN} learned from esteemed professors at {UNIVERSITY}",
"{PERSONAL_PRONOUN} dedicated {POSSESIVE_PRONOUN} time to studies at {UNIVERSITY}",
"{PERSONAL_PRONOUN} developed expertise in {POSSESIVE_PRONOUN} field at {UNIVERSITY}",
"{PERSONAL_PRONOUN} pursued {POSSESIVE_PRONOUN} passion for learning at {UNIVERSITY}",
"{PERSONAL_PRONOUN} engaged in intellectual discourse at {UNIVERSITY}",
"{PERSONAL_PRONOUN} expanded {POSSESIVE_PRONOUN} knowledge through courses at {UNIVERSITY}",
"{PERSONAL_PRONOUN} collaborated with peers and professors at {UNIVERSITY}",
"{PERSONAL_PRONOUN} spent several years studying at {UNIVERSITY}",
"{PERSONAL_PRONOUN} acquired theoretical and practical knowledge at {UNIVERSITY}",
"{PERSONAL_PRONOUN} participated in academic conferences while at {UNIVERSITY}",
"{PERSONAL_PRONOUN} immersed {POSSESIVE_PRONOUN}SELF in campus life at {UNIVERSITY}",
"{PERSONAL_PRONOUN} took advantage of internship opportunities at {UNIVERSITY}",
"{PERSONAL_PRONOUN} refined {POSSESIVE_PRONOUN} analytical skills at {UNIVERSITY}",
"{PERSONAL_PRONOUN} earned a degree from {UNIVERSITY}",
"{PERSONAL_PRONOUN} pursued a major in {POSSESIVE_PRONOUN} chosen discipline at {UNIVERSITY}",
"{PERSONAL_PRONOUN} contributed to student organizations at {UNIVERSITY}",
"{PERSONAL_PRONOUN} received academic accolades at {UNIVERSITY}",
"{PERSONAL_PRONOUN} conducted groundbreaking research at {UNIVERSITY}",
"{PERSONAL_PRONOUN} developed critical thinking skills at {UNIVERSITY}",
"{PERSONAL_PRONOUN} became well-versed in {POSSESIVE_PRONOUN} subject at {UNIVERSITY}",
"{PERSONAL_PRONOUN} gained invaluable insights at {UNIVERSITY}",
"{PERSONAL_PRONOUN} was actively involved in academic discussions at {UNIVERSITY}",
"{PERSONAL_PRONOUN} benefited from state-of-the-art facilities at {UNIVERSITY}",
"{PERSONAL_PRONOUN} enhanced {POSSESIVE_PRONOUN} problem-solving abilities at {UNIVERSITY}",
"{PERSONAL_PRONOUN} explored interdisciplinary studies at {UNIVERSITY}",
"{PERSONAL_PRONOUN} received mentorship and support from professors at {UNIVERSITY}",
"{PERSONAL_PRONOUN} dedicated years to mastering {POSSESIVE_PRONOUN} field at {UNIVERSITY}",
"{PERSONAL_PRONOUN} was a diligent student at {UNIVERSITY}",
"{PERSONAL_PRONOUN} participated in exchange programs through {UNIVERSITY}",
"{PERSONAL_PRONOUN} undertook challenging coursework at {UNIVERSITY}",
"{PERSONAL_PRONOUN} spent countless hours in the library at {UNIVERSITY}",
"{PERSONAL_PRONOUN} thrived in an intellectually stimulating environment at {UNIVERSITY}",
"{PERSONAL_PRONOUN} explored new perspectives through learning at {UNIVERSITY}",
"{PERSONAL_PRONOUN} refined {POSSESIVE_PRONOUN} research skills at {UNIVERSITY}",
"{PERSONAL_PRONOUN} broadened {POSSESIVE_PRONOUN} academic horizons at {UNIVERSITY}",
"{PERSONAL_PRONOUN} was an active member of the academic community at {UNIVERSITY}",
"{PERSONAL_PRONOUN} built lifelong connections at {UNIVERSITY}",
"{PERSONAL_PRONOUN} took on leadership roles in student groups at {UNIVERSITY}",
"{PERSONAL_PRONOUN} pursued {POSSESIVE_PRONOUN} aspirations through education at {UNIVERSITY}",
"{PERSONAL_PRONOUN} embarked on {POSSESIVE_PRONOUN} academic journey at {UNIVERSITY}"
]


major_templates = [
"{PERSONAL_PRONOUN} specialized in {MAJOR} at university",
"{PERSONAL_PRONOUN} pursued a degree in {MAJOR}",
"{PERSONAL_PRONOUN} dedicated {POSSESIVE_PRONOUN} studies to {MAJOR}",
"{PERSONAL_PRONOUN} engaged in extensive coursework in {MAJOR}",
"{PERSONAL_PRONOUN} conducted research in {MAJOR}",
"{PERSONAL_PRONOUN} gained in-depth knowledge of {MAJOR}",
"{PERSONAL_PRONOUN} explored both theoretical and practical aspects of {MAJOR}",
"{PERSONAL_PRONOUN} mastered core principles of {MAJOR}",
"{PERSONAL_PRONOUN} applied critical thinking skills in {MAJOR}",
"{PERSONAL_PRONOUN} built expertise in {MAJOR} through hands-on experience",
"{PERSONAL_PRONOUN} completed advanced studies in {MAJOR}",
"{PERSONAL_PRONOUN} developed technical skills in {MAJOR}",
"{PERSONAL_PRONOUN} earned a degree with a concentration in {MAJOR}",
"{PERSONAL_PRONOUN} was immersed in {MAJOR} during university",
"{PERSONAL_PRONOUN} took specialized courses in {MAJOR}",
"{PERSONAL_PRONOUN} worked on capstone projects related to {MAJOR}",
"{PERSONAL_PRONOUN} honed analytical abilities through {MAJOR}",
"{PERSONAL_PRONOUN} explored interdisciplinary applications of {MAJOR}",
"{PERSONAL_PRONOUN} conducted case studies within {MAJOR}",
"{PERSONAL_PRONOUN} refined problem-solving skills in {MAJOR}",
"{PERSONAL_PRONOUN} collaborated on research projects in {MAJOR}",
"{PERSONAL_PRONOUN} presented findings on {MAJOR} at academic conferences",
"{PERSONAL_PRONOUN} engaged in discussions on contemporary issues in {MAJOR}",
"{PERSONAL_PRONOUN} completed an internship related to {MAJOR}",
"{PERSONAL_PRONOUN} participated in fieldwork for {MAJOR}",
"{PERSONAL_PRONOUN} deepened {POSSESIVE_PRONOUN} understanding of {MAJOR}",
"{PERSONAL_PRONOUN} examined historical developments in {MAJOR}",
"{PERSONAL_PRONOUN} studied under renowned professors in {MAJOR}",
"{PERSONAL_PRONOUN} developed innovative solutions in {MAJOR}",
"{PERSONAL_PRONOUN} explored emerging trends in {MAJOR}",
"{PERSONAL_PRONOUN} gained hands-on experience through lab work in {MAJOR}",
"{PERSONAL_PRONOUN} engaged in data analysis related to {MAJOR}",
"{PERSONAL_PRONOUN} pursued a thesis in {MAJOR}",
"{PERSONAL_PRONOUN} studied the societal impact of {MAJOR}",
"{PERSONAL_PRONOUN} applied mathematical concepts to {MAJOR}",
"{PERSONAL_PRONOUN} contributed to academic publications in {MAJOR}",
"{PERSONAL_PRONOUN} examined ethical implications in {MAJOR}",
"{PERSONAL_PRONOUN} learned industry-standard practices in {MAJOR}",
"{PERSONAL_PRONOUN} developed programming skills relevant to {MAJOR}",
"{PERSONAL_PRONOUN} worked on group projects in {MAJOR}",
"{PERSONAL_PRONOUN} applied theoretical models in {MAJOR}",
"{PERSONAL_PRONOUN} participated in case competitions related to {MAJOR}",
"{PERSONAL_PRONOUN} refined {POSSESIVE_PRONOUN} communication skills through {MAJOR} coursework",
"{PERSONAL_PRONOUN} explored policy implications of {MAJOR}",
"{PERSONAL_PRONOUN} examined real-world applications of {MAJOR}",
"{PERSONAL_PRONOUN} studied foundational texts in {MAJOR}",
"{PERSONAL_PRONOUN} engaged in mentorship programs related to {MAJOR}",
"{PERSONAL_PRONOUN} learned about cross-disciplinary connections to {MAJOR}",
"{PERSONAL_PRONOUN} developed critical perspectives in {MAJOR}",
"{PERSONAL_PRONOUN} expanded {POSSESIVE_PRONOUN} academic horizons through {MAJOR}"
]


employer_templates = [
"{PERSONAL_PRONOUN} was employed at {EMPLOYER}",
"{PERSONAL_PRONOUN} built {POSSESIVE_PRONOUN} career at {EMPLOYER}",
"{PERSONAL_PRONOUN} gained valuable experience at {EMPLOYER}",
"{PERSONAL_PRONOUN} worked as a professional at {EMPLOYER}",
"{PERSONAL_PRONOUN} served in a key role at {EMPLOYER}",
"{PERSONAL_PRONOUN} took on responsibilities at {EMPLOYER}",
"{PERSONAL_PRONOUN} played a vital role at {EMPLOYER}",
"{PERSONAL_PRONOUN} contributed to projects at {EMPLOYER}",
"{PERSONAL_PRONOUN} was part of the team at {EMPLOYER}",
"{PERSONAL_PRONOUN} engaged in professional activities at {EMPLOYER}",
"{PERSONAL_PRONOUN} developed skills while working at {EMPLOYER}",
"{PERSONAL_PRONOUN} spent years working at {EMPLOYER}",
"{PERSONAL_PRONOUN} advanced {POSSESIVE_PRONOUN} career at {EMPLOYER}",
"{PERSONAL_PRONOUN} was a dedicated employee at {EMPLOYER}",
"{PERSONAL_PRONOUN} took part in major initiatives at {EMPLOYER}",
"{PERSONAL_PRONOUN} was an integral part of {EMPLOYER}",
"{PERSONAL_PRONOUN} held a position at {EMPLOYER}",
"{PERSONAL_PRONOUN} pursued {POSSESIVE_PRONOUN} profession at {EMPLOYER}",
"{PERSONAL_PRONOUN} worked on high-impact projects at {EMPLOYER}",
"{PERSONAL_PRONOUN} gained industry knowledge at {EMPLOYER}",
"{PERSONAL_PRONOUN} developed expertise through {EMPLOYER}",
"{PERSONAL_PRONOUN} honed {POSSESIVE_PRONOUN} skills at {EMPLOYER}",
"{PERSONAL_PRONOUN} was a valued team member at {EMPLOYER}",
"{PERSONAL_PRONOUN} made significant contributions to {EMPLOYER}",
"{PERSONAL_PRONOUN} played a key part in operations at {EMPLOYER}",
"{PERSONAL_PRONOUN} achieved professional growth at {EMPLOYER}",
"{PERSONAL_PRONOUN} collaborated with colleagues at {EMPLOYER}",
"{PERSONAL_PRONOUN} worked diligently at {EMPLOYER}",
"{PERSONAL_PRONOUN} held an influential role at {EMPLOYER}",
"{PERSONAL_PRONOUN} delivered results at {EMPLOYER}",
"{PERSONAL_PRONOUN} was involved in strategic planning at {EMPLOYER}",
"{PERSONAL_PRONOUN} managed projects at {EMPLOYER}",
"{PERSONAL_PRONOUN} oversaw critical tasks at {EMPLOYER}",
"{PERSONAL_PRONOUN} thrived in {POSSESIVE_PRONOUN} career at {EMPLOYER}",
"{PERSONAL_PRONOUN} worked to achieve success at {EMPLOYER}",
"{PERSONAL_PRONOUN} established {POSSESIVE_PRONOUN} professional reputation at {EMPLOYER}",
"{PERSONAL_PRONOUN} contributed to the mission of {EMPLOYER}",
"{PERSONAL_PRONOUN} gained hands-on experience at {EMPLOYER}",
"{PERSONAL_PRONOUN} executed major assignments at {EMPLOYER}",
"{PERSONAL_PRONOUN} was recognized for {POSSESIVE_PRONOUN} contributions at {EMPLOYER}",
"{PERSONAL_PRONOUN} had a rewarding career at {EMPLOYER}",
"{PERSONAL_PRONOUN} took on leadership responsibilities at {EMPLOYER}",
"{PERSONAL_PRONOUN} was a key contributor at {EMPLOYER}",
"{PERSONAL_PRONOUN} delivered excellence at {EMPLOYER}",
"{PERSONAL_PRONOUN} brought innovation to {EMPLOYER}",
"{PERSONAL_PRONOUN} supported business goals at {EMPLOYER}",
"{PERSONAL_PRONOUN} was committed to {POSSESIVE_PRONOUN} work at {EMPLOYER}",
"{PERSONAL_PRONOUN} provided expertise at {EMPLOYER}",
"{PERSONAL_PRONOUN} was known for {POSSESIVE_PRONOUN} dedication at {EMPLOYER}",
"{PERSONAL_PRONOUN} played a crucial role in success at {EMPLOYER}"
]
## Employer


employer_city_templates = [
"{PERSONAL_PRONOUN} worked professionally in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} established {POSSESIVE_PRONOUN} career in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} took on professional responsibilities in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} was engaged in work assignments in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} contributed to industry growth in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} advanced {POSSESIVE_PRONOUN} professional journey in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} built {POSSESIVE_PRONOUN} expertise in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} participated in business operations in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} developed professional skills in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} expanded {POSSESIVE_PRONOUN} network while working in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} managed key projects in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} collaborated with industry leaders in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} was actively involved in business activities in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} contributed to innovative solutions in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} provided expertise in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} took on leadership roles in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} played a crucial role in business success in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} handled professional responsibilities in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} navigated the corporate landscape in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} worked with diverse teams in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} pursued professional growth opportunities in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} engaged in entrepreneurial ventures in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} took part in major business initiatives in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} achieved career milestones in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} delivered results for organizations in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} led strategic efforts in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} participated in groundbreaking projects in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} gained valuable insights through work in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} contributed to economic development in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} enhanced {POSSESIVE_PRONOUN} skill set in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} thrived in the work environment of {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} took part in cross-functional teams in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} strengthened {POSSESIVE_PRONOUN} professional profile in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} played a key role in the workforce of {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} built professional relationships in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} collaborated on high-profile projects in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} gained industry recognition through work in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} handled business operations in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} engaged in consulting work in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} pursued career opportunities in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} achieved professional success in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} worked across various sectors in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} developed innovative strategies in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} contributed to corporate growth in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} was part of a thriving work culture in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} applied {POSSESIVE_PRONOUN} expertise to projects in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} supported key business initiatives in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} maintained a strong work presence in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} excelled in {POSSESIVE_PRONOUN} profession in {EMPLOYER_CITY}",
"{PERSONAL_PRONOUN} established a successful work history in {EMPLOYER_CITY}"
]