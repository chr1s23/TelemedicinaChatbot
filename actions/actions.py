from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import ActiveLoop, FollowupAction
from typing import Any, Dict, Text, List, Union
import onnxruntime_genai as og

# Cargar el modelo y el tokenizador desde el contenedor
model = og.Model('/app/cpu_and_mobile/phi3-mini-128k-onnx')
# cargar el modelo desde local
#model = og.Model('/home/admintele/projects/models/phi3-mini-128k-onnx')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

search_options = { 'max_new_tokens': 290, 'temperature': 0.7 }

# Plantilla para el formato del chat
chat_template = '''<|system|>
                    Eres un asistente médico especializado en salud sexual y reproductiva, enfocado en Virus de Papiloma Humano (VPH), ETS, anticoncepción y prevención del cáncer de cuello uterino.
                    Brindas información clara y confiable sobre estos temas.

                    También ofreces información y apoyo a víctimas de **violencia de género, abuso sexual o maltrato**, respondiendo con empatía y sin juicio. Indica la importancia de acudir a **líneas de ayuda, refugios o autoridades**.

                    ### **Reglas de Respuesta**:
                    1. **No das diagnósticos ni recomiendas tratamientos.** Siempre sugiere consultar con un médico.
                    2. **Responde de manera clara, científica y comprensible.** Usa lenguaje sencillo y sin juicios.
                    - **Si la pregunta es sobre métodos anticonceptivos, DEBES DIVIDIRLOS en dos grupos: los que previenen el embarazo y los que protegen contra el VPH. NO PUEDES OMITIR UNO DE LOS GRUPOS.**
                    - Menciónalos en un solo párrafo, sin usar listas ni numeración.
                    3. **Si la pregunta trata sobre violencia sexual, agresión o abuso**:
                    - Responde con empatía, sin culpar a la víctima.
                    - Indica que la persona no está sola y que existen lugares donde puede recibir apoyo y protección.
                    - **Nunca minimices la situación ni ignores la gravedad del problema.**
                    4. **Si la pregunta NO está relacionada con salud sexual, reproductiva o violencia de género, RESPONDE EXCLUSIVAMENTE con el siguiente mensaje y NO intentes responder la pregunta:**
                    *"Lo siento, pero solo puedo brindar información sobre salud sexual y reproductiva, así como apoyo en casos de violencia de género o abuso. ¿Puedo ayudarte con alguna duda sobre estos temas?"*
                    5. **Si no tienes suficiente información sobre una pregunta, indica que el usuario debe consultar con un médico.**
                    6. **Si el usuario está ansioso o preocupado, responde con empatía y tranquilidad.**
                    7. **PROHIBIDO hacer preguntas de seguimiento. No debes generar preguntas después de la respuesta. Si lo haces, la respuesta será eliminada. Solo responde a la consulta sin agregar más información o dudas.**
                    Sigue estas directrices en todas tus respuestas.
                    <|end|>\n<|user|>\n{input} <|end|>\n<|assistant|>'''

# Función para generar respuesta
def generate_response(input_text):
    # Crear el prompt con la pregunta del usuario
    prompt = chat_template.format(input=input_text)

    # Codificar el prompt a tokens
    input_tokens = tokenizer.encode(prompt)

    # Configurar los parámetros del generador
    params = og.GeneratorParams(model)
    #params.set_search_options(**search_options)
    params.input_ids = input_tokens
    # Calcular el límite total de tokens permitidos (entrada + salida)
    total_max_length = len(input_tokens) + search_options.get('max_new_tokens', 250)
    # ✅ Aplicar opciones de búsqueda correctamente
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
        print(response)
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
        print("User:", user_input)    
        response = generate_response(user_input)
        dispatcher.utter_message(text=response)
        return []

class ValidateAutomuestreoForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_automuestreo_form"
    
    async def extract_fecha_ultima_menstruacion(
    self, dispatcher, tracker: Tracker, domain
    ) -> Dict[Text, Any]:
        if tracker.get_slot("requested_slot") == "fecha_ultima_menstruacion":
            user_input = tracker.latest_message.get("text")
            return {"fecha_ultima_menstruacion": user_input}
    # Si no es el momento adecuado, no hagas nada
        return {}
    
    async def extract_fecha_ultimo_pap(
    self, dispatcher, tracker: Tracker, domain
    ) -> Dict[Text, Any]:
        if tracker.get_slot("requested_slot") == "fecha_ultimo_pap":
            user_input = tracker.latest_message.get("text")
            return {"fecha_ultimo_pap": user_input}
    # Si no es el momento adecuado, no hagas nada
        return {}
    
    async def extract_fecha_ultimo_vph(
    self, dispatcher, tracker: Tracker, domain
    ) -> Dict[Text, Any]:
        if tracker.get_slot("requested_slot") == "fecha_ultimo_vph":
            user_input = tracker.latest_message.get("text")
            return {"fecha_ultimo_vph": user_input}
    # Si no es el momento adecuado, no hagas nada
        return {}
    
    async def extract_num_parejas_sexuales(
    self, dispatcher, tracker: Tracker, domain
    ) -> Dict[Text, Any]:
        if tracker.get_slot("requested_slot") == "num_parejas_sexuales":
            user_input = tracker.latest_message.get("text")
            return {"num_parejas_sexuales": user_input}
    # Si no es el momento adecuado, no hagas nada
        return {}
    
    async def extract_nombre_ets(
    self, dispatcher, tracker: Tracker, domain
    ) -> Dict[Text, Any]:
        if tracker.get_slot("requested_slot") == "nombre_ets":
            user_input = tracker.latest_message.get("text")
            return {"nombre_ets": user_input}
    # Si no es el momento adecuado, no hagas nada
        return {}

    async def extract_nombre_enfermedad_autoinmune(
    self, dispatcher, tracker: Tracker, domain
    ) -> Dict[Text, Any]:
        if tracker.get_slot("requested_slot") == "nombre_enfermedad_autoinmune":
            user_input = tracker.latest_message.get("text")
            return {"nombre_enfermedad_autoinmune": user_input}
    # Si no es el momento adecuado, no hagas nada
        return {}

    async def required_slots(
        self,
        domain_slots: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Text]:
        # Empieza con los slots definidos en el domain.yml
        slots_requeridos = domain_slots.copy()

        # Slots adicionales si cumple con los criterios de inclusion y exclusion
        if tracker.get_slot("edad_usuario") is True:
            slots_requeridos.append("es_sexual_activa")

        if tracker.get_slot("es_sexual_activa") is True:
            slots_requeridos.append("relacion_reciente_horas")

        if tracker.get_slot("relacion_reciente_horas") is False:
            slots_requeridos.append("tiene_capacidad_mental")

        if tracker.get_slot("tiene_capacidad_mental") is True:
            slots_requeridos.append("habla_espanol")

        if tracker.get_slot("habla_espanol") is True:
            slots_requeridos.append("esta_embarazada")

        if tracker.get_slot("esta_embarazada") is False:
            slots_requeridos.append("tratamiento_previo_ccu")

        if tracker.get_slot("tratamiento_previo_ccu") is False:
            slots_requeridos.append("toma_medi_intravaginal")

        if tracker.get_slot("toma_medi_intravaginal") is False:
            slots_requeridos.append("esta_menstruando_ahora")

        if tracker.get_slot("esta_menstruando_ahora") is False:
            slots_requeridos.append("fecha_ultima_menstruacion")
            slots_requeridos.append("fecha_ultimo_pap")
            slots_requeridos.append("fecha_ultimo_vph")
            slots_requeridos.append("num_parejas_sexuales")
            slots_requeridos.append("tiene_ets")
            slots_requeridos.append("tiene_enfermedad_autoinmune")

        if tracker.get_slot("tiene_ets") is True:
            slots_requeridos.append("nombre_ets")

        if tracker.get_slot("tiene_enfermedad_autoinmune") is True:
            slots_requeridos.append("nombre_enfermedad_autoinmune")

        return slots_requeridos

    def validate_edad_usuario(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is True:
            return {"edad_usuario": True}
        else:
            dispatcher.utter_message(response="utter_no_cumple_edad")
            return {"formulario_completo": False}
        
    def validate_es_sexual_activa(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is True:
            return {"es_sexual_activa": True}
        else:
            dispatcher.utter_message(response="utter_no_tiene_relacion_sexual")
            return {"formulario_completo": False}
        
    def validate_tiene_capacidad_mental(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is True:
            return {"tiene_capacidad_mental": True}
        else:
            dispatcher.utter_message(response="utter_no_capacidad_mental")
            return {"formulario_completo": False}
        
    def validate_habla_espanol(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is True:
            return {"habla_espanol": True}
        else:
            dispatcher.utter_message(response="utter_no_habla_espanol")
            return {"formulario_completo": False}
        
    def validate_esta_embarazada(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is False:
            return {"esta_embarazada": False}
        else:
            dispatcher.utter_message(response="utter_esta_embarazada")
            return {"formulario_completo": False}
        
    def validate_tratamiento_previo_ccu(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is False:
            return {"tratamiento_previo_ccu": False}
        else:
            dispatcher.utter_message(response="utter_tratamiento_previo_ccu")
            return {"formulario_completo": False}
        
    def validate_relacion_reciente_horas(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is False:
            return {"relacion_reciente_horas": False}
        else:
            dispatcher.utter_message(response="utter_relacion_reciente")
            return {"formulario_completo": False}
        
    def validate_toma_medi_intravaginal(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is False:
            return {"toma_medi_intravaginal": False}
        else:
            dispatcher.utter_message(response="utter_toma_medi_intravaginal")
            return {"formulario_completo": False}
        
    def validate_esta_menstruando_ahora(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is False:
            return {"esta_menstruando_ahora": False}
        else:
            dispatcher.utter_message(response="utter_esta_menstruando_ahora")
            return {"formulario_completo": False}
        
    def validate_fecha_ultima_menstruacion(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is None:
            return {"fecha_ultima_menstruacion": None}
        else:
            return {"fecha_ultima_menstruacion": slot_value}
        
    def validate_fecha_ultimo_pap(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is None:
            return {"fecha_ultimo_pap": None}
        else:
            return {"fecha_ultimo_pap": slot_value}
        
    def validate_fecha_ultimo_vph(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is None:
            return {"fecha_ultimo_vph": None}
        else:
            return {"fecha_ultimo_vph": slot_value}
        
    def validate_num_parejas_sexuales(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if slot_value is None:
            return {"num_parejas_sexuales": None}
        else:
            return {"num_parejas_sexuales": slot_value}
        
    def validate_tiene_enfermedad_autoinmune(self, slot_value: bool, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        return {"formulario_completo": True}
    
class ActionDoNothing(Action):
    def name(self) -> Text:
        return "action_do_nothing"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):
        if tracker.get_slot("formulario_completo") is True:
            dispatcher.utter_message(response="utter_proceso_completo_indicaciones")
        else:
            dispatcher.utter_message(text="Gracias por particiar en el estudio, en este momento no puede realizarte el automuestreo. Sin embargo, es importante que intentes en otra ocasion")
        return []
        