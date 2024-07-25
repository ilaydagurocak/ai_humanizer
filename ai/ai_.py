import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
import random

class AIHumanizer:
    def __init__(self, model_name='gpt2-medium'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.nlp = spacy.load('en_core_web_sm')
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def humanize_text(self, ai_text, max_length=150):
        inputs = self.tokenizer(ai_text, return_tensors='pt', padding=True).to(self.device)
        output = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )
        humanized_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.post_process(humanized_text)

    def post_process(self, text):
        # Remove redundant phrases and improve sentence structure
        text = text.replace('\n', ' ')
        sentences = text.split('. ')
        sentences = [sentence.capitalize() for sentence in sentences if sentence]
        processed_text = '. '.join(sentences)
        if not processed_text.endswith('.'):
            processed_text += '.'
        
        # Additional processing to make the text more human-like
        processed_text = self.refine_text(processed_text)
        
        return processed_text

    def refine_text(self, text):
        # Split text into paragraphs
        paragraphs = text.split('. ')
        refined_paragraphs = []
        
        for paragraph in paragraphs:
            # Further split paragraphs into sentences
            sentences = paragraph.split('. ')
            refined_sentences = []
            
            for sentence in sentences:
                # Remove redundant phrases and refine sentence structure
                if len(sentence) > 10:  # simple check to filter out very short sentences
                    sentence = sentence.replace('..', '.')
                    sentence = sentence.strip()
                    refined_sentences.append(sentence)
            
            refined_paragraphs.append('. '.join(refined_sentences))
        
        # Add personal comments
        personal_comments = self.add_personal_comments(refined_paragraphs)
        
        return '\n\n'.join(personal_comments)

    def add_personal_comments(self, paragraphs):
        # Define topic-specific personal comments
        topic_comments = {
            'technology': [
                "As a tech enthusiast, I find this development fascinating.",
                "This reminds me of my own experiences with new gadgets.",
                "Technology is evolving so quickly; it's hard to keep up!",
                "I'm always excited about new tech trends.",
                "This is a breakthrough in the tech industry."
            ],
            'science': [
                "From my perspective, scientific discoveries always bring excitement.",
                "I believe that science has the potential to solve many of our problems.",
                "Science constantly amazes me with new findings.",
                "I'm fascinated by how science can explain so much about our world.",
                "Scientific research is the key to our future."
            ],
            'history': [
                "As a history buff, I find this period particularly interesting.",
                "This event had a significant impact on the course of history.",
                "History teaches us so much about our present.",
                "I'm always intrigued by historical events like this.",
                "Studying history gives us insight into our future."
            ],
            'health': [
                "Health is a topic close to my heart.",
                "In my opinion, maintaining good health should be a priority for everyone.",
                "I'm always looking for new ways to stay healthy.",
                "Health trends are constantly changing, it's important to stay informed.",
                "A healthy lifestyle is essential for a happy life."
            ],
            'education': [
                "Education is the foundation of a successful society.",
                "I believe that everyone deserves access to quality education.",
                "Education opens so many doors in life.",
                "I'm passionate about lifelong learning.",
                "The education system needs constant improvement."
            ],
            'environment': [
                "Protecting the environment is crucial for our future.",
                "I believe that environmental conservation should be a priority.",
                "I'm always looking for ways to reduce my environmental footprint.",
                "Environmental issues are becoming more urgent every day.",
                "Sustainable practices are essential for the health of our planet."
            ],
            'economy': [
                "The economy affects every aspect of our lives.",
                "I believe that economic stability is crucial for a thriving society.",
                "Economic trends are fascinating to study.",
                "I'm always interested in learning more about economic policies.",
                "The global economy is constantly changing."
            ],
            'politics': [
                "Politics play a huge role in shaping our society.",
                "I believe that everyone should stay informed about political issues.",
                "Political decisions have far-reaching consequences.",
                "I'm always interested in discussing political topics.",
                "The political landscape is constantly evolving."
            ],
            'art': [
                "Art is a powerful form of expression.",
                "I believe that art has the ability to change the world.",
                "I'm always inspired by the creativity of artists.",
                "Art can convey complex emotions and ideas.",
                "The art world is full of fascinating people and stories."
            ],
            'literature': [
                "Literature opens up new worlds and perspectives.",
                "I believe that reading is essential for personal growth.",
                "I'm always looking for new books to read.",
                "Literature can teach us so much about the human experience.",
                "The literary world is full of diverse voices and stories."
            ],
            'travel': [
                "Traveling is one of the best ways to learn about the world.",
                "I believe that travel broadens the mind.",
                "I'm always planning my next adventure.",
                "Travel experiences create lifelong memories.",
                "Exploring new places is always exciting."
            ],
            'food': [
                "Food is a universal language that brings people together.",
                "I believe that trying new foods is an adventure.",
                "I'm always on the lookout for new recipes.",
                "Food has a unique way of connecting cultures.",
                "Exploring different cuisines is one of my favorite pastimes."
            ],
            'sports': [
                "Sports have the power to unite people.",
                "I believe that physical activity is crucial for a healthy life.",
                "I'm always excited to watch my favorite teams play.",
                "Sports teach valuable lessons about teamwork and perseverance.",
                "The world of sports is always full of excitement."
            ],
            'music': [
                "Music is a universal language.",
                "I believe that music has the power to heal.",
                "I'm always discovering new artists and genres.",
                "Music can convey emotions that words cannot.",
                "The music industry is constantly evolving."
            ],
            'fashion': [
                "Fashion is a form of self-expression.",
                "I believe that everyone should find their own style.",
                "I'm always excited to see new fashion trends.",
                "Fashion is about more than just clothes; it's about identity.",
                "The fashion industry is full of creativity."
            ],
            'technology': [
                "Technology is constantly advancing.",
                "I believe that technology has the power to improve our lives.",
                "I'm always intrigued by new tech innovations.",
                "Technology is shaping the future in unimaginable ways.",
                "The tech industry is full of opportunities."
            ],
            'science': [
                "Science is a never-ending journey of discovery.",
                "I believe that scientific research is crucial for progress.",
                "I'm fascinated by the mysteries of the universe.",
                "Science can explain so many phenomena in our world.",
                "The field of science is always evolving."
            ],
            'history': [
                "History is a window into the past.",
                "I believe that understanding history is essential for our future.",
                "I'm always learning new things about historical events.",
                "History helps us understand our present.",
                "The study of history is full of fascinating stories."
            ],
            'health': [
                "Health is wealth.",
                "I believe that taking care of our health is paramount.",
                "I'm always seeking ways to improve my wellbeing.",
                "Health is a lifelong journey.",
                "The field of health and wellness is constantly evolving."
            ],
            'education': [
                "Education is the key to success.",
                "I believe that learning is a lifelong process.",
                "I'm passionate about the value of education.",
                "Education empowers individuals and communities.",
                "The education system needs continuous improvement."
            ],
            'environment': [
                "The environment is our shared responsibility.",
                "I believe that we must take action to protect our planet.",
                "I'm always looking for ways to be more eco-friendly.",
                "Environmental conservation is crucial for our future.",
                "The state of the environment affects us all."
            ],
            'economy': [
                "The economy is a complex system.",
                "I believe that understanding economics is important for everyone.",
                "I'm always interested in economic trends and policies.",
                "Economic stability is crucial for a prosperous society.",
                "The global economy is interconnected."
            ],
            'politics': [
                "Politics influence every aspect of our lives.",
                "I believe that political engagement is crucial for democracy.",
                "I'm always keeping up with political news.",
                "Political decisions have wide-ranging impacts.",
                "The political landscape is always changing."
            ],
            'art': [
                "Art is a reflection of society.",
                "I believe that art can inspire change.",
                "I'm always exploring different art forms.",
                "Art allows us to see the world from different perspectives.",
                "The world of art is full of creativity."
            ],
            'literature': [
                "Literature opens up new worlds.",
                "I believe that reading broadens the mind.",
                "I'm always on the lookout for new books.",
                "Literature can teach us about different cultures and experiences.",
                "The literary world is rich with stories."
            ],
            'travel': [
                "Travel broadens the horizons.",
                "I believe that travel is an essential part of life.",
                "I'm always planning my next trip.",
                "Travel experiences create unforgettable memories.",
                "Exploring new places is always exciting."
            ],
            'food': [
                "Food is a cultural experience.",
                "I believe that trying new foods is an adventure.",
                "I'm always excited to try new recipes.",
                "Food brings people together.",
                "Exploring different cuisines is one of my passions."
            ],
            'sports': [
                "Sports bring people together.",
                "I believe that physical activity is crucial for health.",
                "I'm always excited to watch sporting events.",
                "Sports teach valuable life lessons.",
                "The world of sports is full of excitement."
            ],
            'music': [
                "Music is a universal language.",
                "I believe that music has the power to heal.",
                "I'm always discovering new music.",
                "Music can express emotions that words cannot.",
                "The music industry is ever-changing."
            ],
            'fashion': [
                "Fashion is a form of self-expression.",
                "I believe that fashion reflects individuality.",
                "I'm always keeping up with fashion trends.",
                "Fashion is about more than just clothes.",
                "The fashion industry is innovative."
            ],
            'technology': [
                "Technology is constantly evolving.",
                "I believe that technology can change the world.",
                "I'm fascinated by technological advancements.",
                "Technology is integral to modern life.",
                "The tech industry is full of innovation."
            ],
            'science': [
                "Science is a path to understanding.",
                "I believe that science can solve many problems.",
                "I'm always interested in scientific discoveries.",
                "Science explains the workings of the universe.",
                "The field of science is ever-growing."
            ],
            'history': [
                "History teaches us about our past.",
                "I believe that history is crucial for understanding the present.",
                "I'm always learning from historical events.",
                "History shapes our identity.",
                "The study of history is enlightening."
            ],
            'health': [
                "Health is the foundation of life.",
                "I believe in prioritizing health above all.",
                "I'm always learning about health and wellness.",
                "Health is a continuous journey.",
                "The health field is dynamic and evolving."
            ],
            'education': [
                "Education is the key to opportunity.",
                "I believe in the power of education.",
                "I'm passionate about lifelong learning.",
                "Education transforms lives.",
                "The education system is essential for progress."
            ],
            'environment': [
                "The environment is our collective responsibility.",
                "I believe in taking action to protect the environment.",
                "I'm committed to sustainable living.",
                "Environmental conservation is vital.",
                "The state of the environment impacts everyone."
            ],
            'economy': [
                "The economy is the backbone of society.",
                "I believe in understanding economic principles.",
                "I'm always interested in economic developments.",
                "Economic health is crucial for a thriving society.",
                "The global economy is complex and interconnected."
            ],
            'politics': [
                "Politics shape our world.",
                "I believe in staying informed about political issues.",
                "I'm engaged in political discussions.",
                "Political decisions have significant consequences.",
                "The political climate is constantly shifting."
            ],
            'art': [
                "Art reflects culture.",
                "I believe that art can inspire change.",
                "I'm passionate about exploring art.",
                "Art provides new perspectives.",
                "The art world is diverse and dynamic."
            ],
            'literature': [
                "Literature expands our horizons.",
                "I believe in the transformative power of literature.",
                "I'm always reading new books.",
                "Literature connects us to different experiences.",
                "The literary world is vast and varied."
            ],
            'travel': [
                "Travel enriches the soul.",
                "I believe in the value of travel.",
                "I'm constantly planning my next trip.",
                "Travel experiences are invaluable.",
                "Exploring new places is a passion."
            ],
            'food': [
                "Food is a journey of discovery.",
                "I believe in the joy of trying new foods.",
                "I'm always experimenting with new recipes.",
                "Food brings people together.",
                "Exploring cuisines is a delightful experience."
            ],
            'sports': [
                "Sports foster unity.",
                "I believe in the importance of physical activity.",
                "I'm enthusiastic about watching sports.",
                "Sports teach perseverance and teamwork.",
                "The sports world is thrilling."
            ],
            'music': [
                "Music transcends boundaries.",
                "I believe in the healing power of music.",
                "I'm always exploring new music genres.",
                "Music communicates emotions beautifully.",
                "The music industry is ever-evolving."
            ],
            'fashion': [
                "Fashion expresses individuality.",
                "I believe in the creativity of fashion.",
                "I'm always keeping up with fashion trends.",
                "Fashion is a form of art.",
                "The fashion industry is innovative and exciting."
            ]
        }
        
        refined_paragraphs_with_comments = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                doc = self.nlp(paragraph)
                topics = set()
                for token in doc:
                    if token.lemma_ in topic_comments:
                        topics.add(token.lemma_)
                
                if topics:
                    comment = random.choice(topic_comments[next(iter(topics))])
                    paragraph = f"{paragraph}. {comment}"
                    
                refined_paragraphs_with_comments.append(paragraph)
        
        return refined_paragraphs_with_comments

# Kullanıcıdan metin alıp dönüştüren kısım
if __name__ == "__main__":
    ai_text = input("Lütfen dönüştürmek istediğiniz AI tarafından üretilmiş metni girin: ")
    humanizer = AIHumanizer()
    humanized_text = humanizer.humanize_text(ai_text)
    print("Orijinal AI Metni: ", ai_text)
    print("İnsan Tarafından Yazılmış Gibi Metin: ", humanized_text)
