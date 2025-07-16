from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted
import onnxruntime_genai as og

# Cargar el modelo y el tokenizador desde el contenedor
model = og.Model('/app/cpu_and_mobile/phi3-mini-128k-onnx')
# cargar el modelo desde local
#model = og.Model('/home/admintele/projects/models/phi3-mini-128k-onnx')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

search_options = { 'max_new_tokens': 290, 'temperature': 0.7 }

# Prompt plantilla para el modelo preentenado
chat_template = '''<|system|>
        Eres un asistente especializado en salud sexual y reproductiva. Tu objetivo es brindar información clara, respetuosa y basada en evidencia sobre:
        1. Virus del Papiloma Humano (VPH)
        2. anticoncepción
        3. prevención del cáncer de cuello uterino (CCU)
        4. VIH (Virus de Inmunodeficiencia Humana)
        5. higiene íntima (incluyendo recomendaciones sobre productos de limpieza, jabones íntimos, cuidado personal y prácticas saludables)
        6. relaciones sexuales
        7. infecciones de transmisión sexual (ITS/ETS)
        8. creencias comunes sobre sexualidad
        9. salud sexual en distintas etapas de la vida (adolescencia, adultez, menopausia, etc.)
        
        También puedes ofrecer orientación general y contención emocional a personas que mencionen situaciones de violencia de género o abuso, siempre desde una postura empática, sin juicios, y respetuosa.
        
        Respondes siempre de manera comprensible, científica y empática en MÁXIMO DE 250 CARACTERES. No realizas diagnósticos médicos ni prescribes tratamientos específicos. Siempre que sea necesario, debes sugerir acudir a un profesional de la salud.

        Si la pregunta NO está relacionada directamente con los temas anteriores, RESPONDE EXCLUSIVAMENTE con el siguiente mensaje y NO intentes responder la pregunta:
        **Lo siento, pero solo puedo brindar información sobre salud sexual y reproductiva, así como apoyo en casos de violencia de género o abuso. ¿Puedo ayudarte con alguna duda sobre estos temas?**
<|end|>\n<|user|>\n{input} <|end|>\n<|assistant|>'''

def generate_response(input_text):
    prompt = chat_template.format(input=input_text)

    # Codificar el prompt a tokens
    input_tokens = tokenizer.encode(prompt)

    # Configurar los parámetros del generador
    params = og.GeneratorParams(model)

    params.input_ids = input_tokens
    # Calcular el límite total de tokens permitidos (entrada + salida)
    total_max_length = len(input_tokens) + search_options.get('max_new_tokens', 250)
    # Aplicar opciones de búsqueda correctamente
    params.set_search_options(
        max_length=total_max_length,
        temperature=search_options.get('temperature', 0.5)
    )
    generator = og.Generator(model, params)

    # Generar respuesta
    response = ''
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            response += tokenizer_stream.decode(new_token)
    except KeyboardInterrupt:
        print("\n  --control+c pressed, aborting generation--")
    finally:
        del generator

    return response

class ActionGenerateDetailedResponse(Action):
    def name(self) -> str:
        return "action_consult_info"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_input = tracker.latest_message.get("text")
        response = generate_response(user_input)
        dispatcher.utter_message(text=response)
        return []
    
class ActionFallback(Action):
    def name(self):
        return "action_default_fallback"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_input = tracker.latest_message.get("text")
        response = generate_response(user_input)
        dispatcher.utter_message(text=response)
        return [UserUtteranceReverted()]
        