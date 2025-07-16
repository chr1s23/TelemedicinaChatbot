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

search_options = { 'max_new_tokens': 400, 'temperature': 0.7 }

# Prompt plantilla para el modelo preentenado
chat_template = '''<|system|>
Eres un asistente experto en salud sexual y reproductiva. Responde con información clara, empática y basada en evidencia sobre temas como VPH, anticoncepción, VIH, higiene íntima, ITS, relaciones sexuales y violencia de género. Máximo 400 caracteres.
Si la pregunta no está relacionada, responde:
*Lo siento, solo puedo responder sobre salud sexual y reproductiva, o brindar apoyo en casos de violencia de género. ¿Te gustaría preguntar sobre eso?*
<|end|>\n<|user|>\n{input} <|end|>\n<|assistant|>'''

def generate_response(input_text):
    prompt = chat_template.format(input=input_text)

    print("\n=== PROMPT ENVIADO AL MODELO ===\n")
    print(prompt)
    print("\nTOKEN COUNT:", len(tokenizer.encode(prompt)))
    print("\n===============================\n")

    # Codificar el prompt a tokens
    input_tokens = tokenizer.encode(prompt)

    # Configurar los parámetros del generador
    params = og.GeneratorParams(model)

    params.input_ids = input_tokens
    # Calcular el límite total de tokens permitidos (entrada + salida)
    total_max_length = len(input_tokens) + search_options.get('max_new_tokens', 400)
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

    print("RESPUESTA GENERADA:")
    print(response)
    print("LONGITUD RESPUESTA (caracteres):", len(response))

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
        