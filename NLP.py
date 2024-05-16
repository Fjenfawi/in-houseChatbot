import pandas as pd
import spacy
from neuralNetwork import *
df = pd.read_csv("CQFinal.csv")
category = df.category
category_to_number = {'Login Issues':0
                      ,'forgot password' : 1
                      ,'registration issues':2
                      ,'payment issues':3
                      ,'final grades':4
                      ,'Transcript':5
                      ,'submitting error':6
                      ,'Online Learning':7
                      ,'Library issues':8
                      ,'contact IT':9
                      ,'wifi issues':10
                      ,'Mobile app':11
                      ,'apply for graduation':12
                      ,'final grades':13
                      ,'drop course':14
                      ,'class info ':15
                      ,'Registeration Issue':16
                      ,'Vpn':17
                      ,'guests policy':18
                      ,'change contact info ':19
                      ,'E-mail Issue':20
                      ,'Security and Privacy':21}
df['labeled_category'] = list(map(lambda x: category_to_number[x], category))
model= Model()
nlp= spacy.load("en_core_web_lg")
def process_text(text):
    # Parse the text with SpaCy
    doc = nlp(text)
    # Remove stop words and convert to lowercase
    processed_text = [token.text.lower() for token in doc if not token.is_stop]
    # Join the processed tokens back into a string
    return ' '.join(processed_text)

# Apply the process_text function to the 'text' column and save the changes back
#df['Questions_changed'] = df['Question'].apply(process_text)


df['Vector'] = df['Question'].apply(lambda Question : nlp(Question).vector)



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    df['Vector'].values,
    df['labeled_category'],
    test_size=0.2,
    random_state = 2022,
    stratify = df['labeled_category'] 
)
'''
import numpy as np


# Assuming X_train and X_test are your numpy arrays
# Flatten all elements in X_train
X_train = np.array([np.array(x).flatten() for x in X_train])

# Flatten all elements in X_test
X_test = np.array([np.array(x).flatten() for x in X_test])





X_train= (X_train.reshape(X_train.shape[0],-1).astype(np.float32))
X_test =  (X_test.reshape(X_test.shape[0],-1).astype(np.float32))
'''
'''
model.add(Layer_Dense(X_train.shape[1],256))
model.add(Activation_Relu())
model.add(Layer_Dense(256,256))
model.add(Activation_Relu())
model.add(Layer_Dense(256,23))
model.add(Activation_Softmax())
model.set(
    loss=Loss_categoricalCrossEntropy(),
    optimizer= optimizer_Adam(decay=1e-4),
    accuracy = Accuracy_Categorical())
model.finalize()
model.train(X_train,y_train,validation_data=(X_test,y_test),epochs=10000,batch_size=512,print_every=100)
parameters = model.get_parameters()
model.save('saved_model/s16')
'''

#-----------------------------------


'''
model=Model()
model.add(Layer_Dense(X_train.shape[1],64))
model.add(Activation_Relu())
model.add(Layer_Dense(64,64))
model.add(Activation_Relu())
model.add(Layer_Dense(64,23))
model.add(Activation_Softmax())
model.set(
    loss=Loss_categoricalCrossEntropy(),
    accuracy = Accuracy_Categorical() )


model.finalize()
model.evaluate(X_test,y_test)
model.load_parameters('saved_model/s4')
model.evaluate(X_test,y_test)
'''


model = Model()
userinput = input("how can I help you with")
df['processes_input'] = userinput 
processed_input1 = df['processes_input'].apply(process_text)
processed_input2 = processed_input1[0]
vector_question = nlp(processed_input2).vector
model= Model.load('saved_model/s16')
pre =model.predict(vector_question)
pre2 = model.output_layer_activation.predictions(pre)
print(pre2)
category = next(key for key, value in category_to_number.items() if value == pre2[0])
print(category)
# Example dictionary mapping categories to answers
category_answers = {
    "Login Issues": '''If you're having trouble accessing Moodle, the American University of Kuwait's Information Technology Department can provide assistance They offer a range of IT support services, 
                    including troubleshooting software issues and facilitating the resolution of technical problems 1 .You can reach out to the IT Helpdesk through email,
                    online ticketing systems, or in-person visits to the Helpdesk location 1 . Unfortunately, the specific contact details are not provided in the retrieved document
                    If you're unable to resolve the issue in time and have an assignment due, you can email the completed assignment to your professor before the deadline.
                    This will document that you completed the work on time but had a technology issue 2 3 .Remember to upload the assignment to Moodle once the issue is resolved 2 3 .
                    For further assistance, you may need to contact the IT Department directly at the American University of Kuwait.''',
    "forgot password": '''If you've forgotten your password, you can reset it using the password reset feature provided by the American University of Kuwait. Here are the steps to do so:
                        Enroll for password reset using the URL https://adselfservice.auk.edu.kw/pwdreset 1 .
                        Provide your username and password for enrolling 1 .
                        Once the enrollment is done, you will be able to change/reset your passwords 1 .
                        Alternatively, you can use the Password Reset App - AdselfService Plus for resetting passwords 1 .
                        If you encounter any issues during this process, you can report it to the Information Technology Department via email at ithelpdesk@auk.edu.kw 1
                        . They will investigate the reported violations and escalate them appropriately to higher channels 1''',
    "registration issues": '''If you're having trouble adding courses to your schedule, you should first verify the accuracy of your course schedules on the self-service throughout the semester in which you are enrolled 1 .

                            If you're still encountering issues, it's recommended to seek advice from your academic advisors 1 .
                            They can provide guidance on course selection and help troubleshoot any issues you might be experiencing with the scheduling system.
                            If the problem persists, you can reach out to the Information Technology Department for technical assistance.
                            They provide a wide range of IT support services, including troubleshooting software issues and facilitating the resolution of technical problems 2 .
                            You can access IT Helpdesk support through email, online ticketing systems, and in-person visits to the Helpdesk location 2 .
                            If the issue is related to course availability or specific course requirements, you may need to contact the relevant academic department or the Office of the Registrar for further assistance.
                            ''',
    "payment issues": '''If you're experiencing payment issues, ensure your payment method is valid. For fee-related concerns, contact the finance office for clarification. Email:"finance@auk.edu.kw"''',
    "final grades": '''Grades are typically available a few days after exams. If you believe there's an error, reach out to the professor. Transcript requests can be made through the student portal.''',
    "Transcript": '''If you want to access the unofficial transcript follow these steps: 1- Visit self service website "ssb-prod.ec.auk.edu.kw". 2- go to student services 3-student records 4-View Your Transfer Credit/Academic Transcript''',
    "submitting error": '''Check the file format and size requirements. If the problem persists, contact your instructor or the IT help desk.''',
    "Online Learning": '''Ensure your internet connection is stable. For virtual classes, check the schedule, and if the issue continues, contact the e-learning support team.''',
    'Library issues':'''If you're having trouble accessing e-books and online resources from the university library, the American University of Kuwait's Information Technology Department can provide assistance. They offer a range of IT support services, including troubleshooting software issues and facilitating the resolution of technical problems 1 .

                        You can reach out to the IT Helpdesk through email, online ticketing systems, or in-person visits to the Helpdesk location 1 . They can assist with issues related to accessing online resources, including university-provided computing devices, software applications, email, network connectivity, and more 1 .

                        If the issue persists, you may need to contact the University Library directly. The library staff coordinates collection development, cataloging, and the utilization of print and electronic resources 2 . They can provide academic support to help you identify, locate, and use the library resources 2 .

                        Remember, it's important to resolve these issues as soon as possible to ensure you can access the necessary resources for your academic progress.''',
    'contact IT':'''For technical support related to portal issues at the American University of Kuwait (AUK),
                    you can contact the IT Helpdesk. You can reach them via email at ithelpdesk@auk.edu.kw 1 .
                    If you are facing issues in the classroom, you can also contact them via telephone support on 22299010 1 . Additionally, the Information Technology Department provides walk-in support during university official working hours. You can visit them on the Ground Floor, Science Building '''
    ,'wifi issues':'''For WiFi or other technical issues at the American University of Kuwait (AUK),
                        you can contact the IT Helpdesk. You can reach them via email at ithelpdesk@auk.edu.kw
                        . If you are on campus, you can also visit the Information Technology Department
                        for walk-in support during university official working hours. They are located on the Ground Floor, Science Building'''
     ,'Mobile app':'''yes, you can download mooodle from apple store, or playstore '''
    ,'apply for graduation':'''To apply for graduation at the American University of Kuwait (AUK), you need to submit a completed
                                application for graduation via the Self-Service Graduation Application .
                                After the application has been filed, the Office of the Registrar conducts a degree audit and
                                informs you and your advisor of the remaining requirements via DegreeWorks 1 .
                                It's important to ensure that you have met all degree requirements for graduation, including curriculum and cumulative GPA requirements 1 .
                                The application for graduation must be made by the deadline, which can be found on the Registrar’s webpage'''
    ,'final grades':'''Grades are typically available a few days after exams. If you believe there's an error, reach out to the professor. Transcript requests can be made through the student portal.'''
    ,'drop course':'''To drop a course at the American University of Kuwait (AUK),
                        you can use the "Undergraduate Single Course Drop/Add Form".
                        This form allows you to drop a course you have registered for.
                        Please note that there are specific deadlines for dropping courses,
                        which can be found on the AUK website or academic calendar 2 3 .
                        It's also recommended to consult the tuition refund schedule before withdrawing
                        from a course 2 3 . If you are a scholarship student, you may need to settle the payment
                        for the dropped course, and you should contact the AUK Scholarship & Financial Aid Office
                        at scholarship@auk.edu.kw for further assistance'''
    ,'class info ':'''To get information about your class at the American University of Kuwait (AUK),
                    you can check the course syllabus provided by your instructor.
                    The syllabus typically includes details about the course content, required textbooks,
                    grading scale, and evaluation methods 1 .
                    For online resources, you can visit the university's Learning Management System (LMS)
                    at https://lms.auk.edu.kw/ 1 .
                    If you need further assistance, you can contact the Information Technology Department via email
                    at ithelpdesk@auk.edu.kw 2 . They can help you with technical issues related to accessing
                    online resources'''
    ,'Registeration Issue':'''If you're having trouble adding courses to your schedule, you should first verify the accuracy of your course schedules on the self-service throughout the semester in which you are enrolled 1 .

                            If you're still encountering issues, it's recommended to seek advice from your academic advisors 1 .
                            They can provide guidance on course selection and help troubleshoot any issues you might be experiencing with the scheduling system.
                            If the problem persists, you can reach out to the Information Technology Department for technical assistance.
                            They provide a wide range of IT support services, including troubleshooting software issues and facilitating the resolution of technical problems 2 .
                            You can access IT Helpdesk support through email, online ticketing systems, and in-person visits to the Helpdesk location 2 .
                            If the issue is related to course availability or specific course requirements, you may need to contact the relevant academic department or the Office of the Registrar for further assistance.
                            '''
    ,'Vpn':'''The retrieved documents do not provide specific information on whether the use of a VPN
        (Virtual Private Network) is allowed for accessing resources at the American University of Kuwait (AUK).
        For this information, you may want to contact the Information Technology Department at AUK.
        They can be reached via email at ithelpdesk@auk.edu.kw'''
    ,'guests policy':'''Yes, you are allowed to bring guests to the American University of Kuwait (AUK).
                        However, there are certain protocols and guidelines that need to be followed.
                        All visitors to AUK must register at any of the gates where they will receive a visitor’s
                        badge. Visitors must present some form of valid personal photo ID and state their
                        destination prior to receiving the visitor’s badge'''
    ,'change contact info ':'''The retrieved documents do not provide specific information on how to change your
                    contact information at the American University of Kuwait (AUK).
                    For this information, you may want to contact the Registrar's Office or the Information
                    Technology Department at AUK. The Information Technology Department can be reached via email
                    at ithelpdesk@auk.edu.kw'''
    ,'E-mail Issue':'''For email-related issues at the American University of Kuwait (AUK),
                    you can contact the IT Helpdesk. You can reach them via email at ithelpdesk@auk.edu.kw.
                    They can assist with troubleshooting and resolving technical issues related to the university's
                    email system. If you are on campus, you can also visit the Information Technology Department
                    for walk-in support during university official working hours. They are located on the Ground
                    Floor, Science Building'''
    ,'Security and Privacy':'''To create a strong password for your university accounts at the American University
                            of Kuwait (AUK), you should follow these guidelines:
                            Your password should have a minimum of eight characters 1 .
                            It should include a combination of uppercase letters, lowercase letters, numbers, and symbols 1 .
                            Uppercase characters (A through to Z) 1
                            Lowercase characters (a through to z) 1
                            Numerical digits (0 through to 9) 1
                            Symbols characters (e.g.! $ # % @ +) 1
                            Avoid easily guessable information, such as birthdays, names, or dictionary words 1 .
                                Previous passwords used for a university system must not be re-used 1 .
                                For additional security, the university also encourages the use of multi-factor authentication (MFA) for accessing IT resources, systems, and accounts 1 .

                                Remember, you are responsible for protecting your passwords from unauthorized access. This includes not sharing passwords with others, not writing passwords down or storing them in easily accessible locations, not using the same password across multiple accounts,
                                and changing passwords immediately if they suspect unauthorized access or compromise'''
}



# Print the answer based on the predicted category
if category in category_answers:
    print(category_answers[category])
else:
    print("I'm sorry, I don't have an answer for that category.")


