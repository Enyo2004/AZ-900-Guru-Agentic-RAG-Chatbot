from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class AgenticCoding():
    """AgenticCoding crew"""

    # provide the route of the yaml files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def interpreter(self) -> Agent:
        return Agent(
            config=self.agents_config['interpreter'], # type: ignore[index]
            verbose=True
        )

    @agent
    def backend_developer(self) -> Agent:
        return Agent(
            config=self.agents_config['backend_developer'], # type: ignore[index]
            verbose=True
        )

    @agent
    def frontend_developer(self) -> Agent:
        return Agent(
            config=self.agents_config['frontend_developer'], # type: ignore[index]
            verbose=True,
            allow_code_execution=True,
            code_execution_mode='safe'
        )


    @task
    def interpret_the_code(self) -> Task:
        return Task(
            config=self.tasks_config['interpret_the_code'], # type: ignore[index]
        )

    @task
    def python_logic_code(self) -> Task:
        return Task(
            config=self.tasks_config['python_logic_code'], # type: ignore[index]
        )

    @task
    def gradio_code(self) -> Task:
        return Task(
            config=self.tasks_config['gradio_code'], # type: ignore[index]
            output_file='Azure_Rag.py'
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the AgenticCoding crew"""

        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
